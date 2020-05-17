import transformers

from transformers import BertModel, BertConfig
from transformers import ReformerModel, ReformerConfig, Trainer, TrainingArguments
from torch import nn
from torch.nn import functional as F


class AudioTransformerHF(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, num_classes):
        super(AudioTransformerHF, self).__init__()
        self.config = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            intermediate_size=dim_feedforward,
            num_attention_heads=nhead
        )
        self.encoder = BertModel(self.config)
        self.decoder = LinearClassifier(d_model, num_classes)

    def forward(self, x):
        x = self.encoder.forward(inputs_embeds=x)
        x = self.decoder(x)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=10e-12)
        self.decoder = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class SimpleLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(SimpleLinearClassifier, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=10e-12)
        self.decoder = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

