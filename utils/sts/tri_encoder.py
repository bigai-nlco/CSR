import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import cosine_similarity
from .utils import *
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, AutoModel
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

def concat_features(*features):
    return torch.cat(features, dim=0) if features[0] is not None else None


class TriEncoderForClassification_(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
        ).base_model
        self.layer_score = -1
        self.triencoder_head = config.triencoder_head
        classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
                )
        else:
            self.transform = None
        self.condition_transform = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        if self.triencoder_head == 'concat':
            self.concat_transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                ACT2FN[config.hidden_act],
            )
        elif self.triencoder_head == 'hadamard':
            self.concat_transform = None
        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.num_labels == 1:
            self.reshape_function = lambda x: x.reshape(-1)
            if config.objective == 'mse':
                self.loss_fct_cls = nn.MSELoss
                self.loss_fct_kwargs = {}
            elif config.objective in {'triplet', 'triplet_mse'}:
                self.loss_fct_cls = QuadrupletLoss
                self.loss_fct_kwargs = {'distance_function': lambda x, y: 1.0 - cosine_similarity(x, y)}
            else:
                raise ValueError('Only regression and triplet objectives are supported for TriEncoderForClassification')
        else:
            self.reshape_function = lambda x: x.reshape(-1, config.num_labels)
            self.loss_fct_cls = nn.CrossEntropyLoss
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        position_ids_2=None,
        head_mask_2=None,
        inputs_embeds_2=None,
        input_ids_3=None,
        attention_mask_3=None,
        token_type_ids_3=None,
        position_ids_3=None,
        head_mask_3=None,
        inputs_embeds_3=None,
        labels=None,
        **kwargs,
        ):
        bsz, seq_length = input_ids.shape
        input_ids = concat_features(input_ids, input_ids_2)
        attention_mask = concat_features(attention_mask, attention_mask_2)
        token_type_ids = concat_features(token_type_ids, token_type_ids_2)
        position_ids = concat_features(position_ids, position_ids_2)
        head_mask = concat_features(head_mask, head_mask_2)
        inputs_embeds = concat_features(inputs_embeds, inputs_embeds_2)

        conditions = self.backbone(
            input_ids=input_ids_3,
            attention_mask=attention_mask_3,
            token_type_ids=token_type_ids_3,
            position_ids=position_ids_3,
            head_mask=head_mask_3,
            inputs_embeds=inputs_embeds_3,
        ).last_hidden_state
        
        attention_mask = torch.cat([attention_mask_3.repeat(2, 1), attention_mask], dim=1)
        attention_mask_ = self.manip_attention_mask(attention_mask, seq_length)
        conditions = conditions.repeat(2, 1, 1)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask_,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            conditions=conditions,
            output_token_scores=True
        )
        
        features = self.pooler(attention_mask, outputs)
        features_1, features_2 = torch.split(features, bsz, dim=0)
        features_3 = conditions[-1][:, 0]

        loss = None
        if self.transform is not None:
            features_1 = self.transform(features_1)
            features_2 = self.transform(features_2)
        if self.triencoder_head == 'concat':
            features_1 = torch.cat([features_1, features_3], dim=-1)
            features_2 = torch.cat([features_2, features_3], dim=-1)
            features_1 = self.concat_transform(features_1)
            features_2 = self.concat_transform(features_2)
        elif self.triencoder_head == 'hadamard':
            features_1 = features_1 * features_3
            features_2 = features_2 * features_3
        if self.config.objective in {'triplet', 'triplet_mse'}:
            positive_idxs = torch.arange(0, features_1.shape[0]//2)
            negative_idxs = torch.arange(features_1.shape[0]//2, features_1.shape[0])
            positives1 = features_1[positive_idxs]
            positives2 = features_2[positive_idxs]
            negatives1 = features_1[negative_idxs]
            negatives2 = features_2[negative_idxs]
            if labels is not None:
                loss = self.loss_fct_cls(**self.loss_fct_kwargs)(positives1, positives2, negatives1, negatives2)
            logits = cosine_similarity(features_1, features_2, dim=1)
            if self.config.objective == 'triplet_mse' and labels is not None:
                loss += nn.MSELoss()(logits, labels)
            else:
                logits = logits.detach()
        else:
            logits = cosine_similarity(features_1, features_2, dim=1)
            if labels is not None:
                loss = self.loss_fct_cls(**self.loss_fct_kwargs)(logits, labels)

        return ConditionEncoderOutput(
            loss=loss,
            logits=logits,
            token_scores=outputs.token_scores[self.layer_score][:bsz],
            token_scores_2=outputs.token_scores[self.layer_score][bsz:],
        )
    
    def manip_attention_mask(self, mask, qlen=None):
        slen = mask.shape[1]
        qlen = slen if qlen is None else qlen

        mask_expanded = mask.unsqueeze(1)
        mask_3d = mask_expanded.repeat(1, qlen, 1)
        return mask_3d
