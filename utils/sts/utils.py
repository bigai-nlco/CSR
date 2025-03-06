import torch
from torch import nn
import torch
import torch.utils.checkpoint
from torch import nn
from typing import Tuple, Optional
from dataclasses import dataclass
from transformers.utils import ModelOutput

# Pooler class. Copied and adapted from SimCSE code
class Pooler(nn.Module):
    '''
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    '''
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last', 'routing'], 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, attention_mask, outputs=None, last_hidden=None, pooler_output=None, hidden_states=None, pooler_type=None):
        if outputs is not None:
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
        pooler_type = self.pooler_type if pooler_type is None else pooler_type

        if pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif pooler_type == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1) + 1e-10)
        elif pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

@dataclass
class ConditionEncoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    token_scores: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_scores_2: Optional[Tuple[torch.FloatTensor, ...]] = None
    
@dataclass
class MyEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_scores: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class MyModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_scores: Optional[Tuple[torch.FloatTensor, ...]] = None

class QuadrupletLoss:
    def __init__(self, distance_function, margin=1.0):
        'A cosine distance margin quadruplet loss'
        self.margin = margin
        self.distance_function = distance_function

    def __call__(self, pos1, pos2, neg1, neg2):
        dist_pos = self.distance_function(pos1, pos2)
        dist_neg = self.distance_function(neg1, neg2)
        loss = torch.clamp_min(self.margin + dist_pos - dist_neg, 0)
        return loss.mean()

  
class InfoNCELoss:
    def __init__(self, temperature=0.7, distance_function=None):
        'A cosine distance margin quadruplet loss'
        self.temperature = temperature
        self.distance_function = distance_function

    def __call__(self, pos1, pos2, neg1, neg2):
        dist_pos = self.distance_function(pos1, pos2)
        dist_neg1 = self.distance_function(pos1, neg2)
        dist_neg2 = self.distance_function(pos2, neg1)

        dist_pos = torch.exp(dist_pos / self.temperature)
        dist_neg1 = torch.exp(dist_neg1 / self.temperature)
        dist_neg2 = torch.exp(dist_neg2 / self.temperature)
        loss = -torch.log(dist_pos / (dist_neg1 + dist_neg2))
        return loss.mean()

    
class RankingLoss:
    def __init__(self, margin=0.1):
        self.margin = margin

    def __call__(self, token_scores, labels, masks=None):
        total_loss = 0.0
        valid_batch = 0
        for token_score, label, mask in zip(token_scores, labels, masks):
            key_num = torch.sum(label)
            if key_num == 0:
                continue
            else:
                valid_batch += 1

            token_score = token_score[mask==1]
            label = label[mask==1]
            
            positive_indices = label.nonzero(as_tuple=True)
            negative_indices = (label == 0).nonzero(as_tuple=True)

            positive_score = token_score[positive_indices]
            negative_score = token_score[negative_indices]

            differences = negative_score.view(-1, 1) - positive_score.view(1, -1)
            loss = torch.clamp(differences + self.margin, min=0)
            total_loss += torch.mean(loss)
        if valid_batch == 0:
            return torch.tensor(0.0, dtype=token_scores.dtype, device=token_scores.device)
        return total_loss / valid_batch
    