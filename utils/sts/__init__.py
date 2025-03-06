from .modeling_router import RobertaSelfAttention, RobertaAttention, RobertaLayer, RobertaModel
from transformers.models.roberta import modeling_roberta

modeling_roberta.RobertaSelfAttention = RobertaSelfAttention
modeling_roberta.RobertaAttention = RobertaAttention
modeling_roberta.RobertaLayer = RobertaLayer
modeling_roberta.RobertaModel = RobertaModel