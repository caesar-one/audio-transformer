import time
import math

import h5py
import torch
import torch.nn as nn
#import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from raw_transformer.dataset import AudioDataset#, get_proper_shape
from raw_transformer import dataset
from raw_transformer.models import AudioTransformer
from sklearn.metrics import classification_report
from glob import glob

# Trains a single epoch with hyper-parameters provided
def train(model, criterion, optimizer, data, device):
    model.train() # Turn on the train mode
    total_loss = 0.
    progress = tqdm(data, "Training")
    for X, y in progress:
        X = X.permute((1,0,2)).to(device) # convert to seq first shape and move to desired device
        y = y.long()
        optimizer.zero_grad()
        output = model(X)
        output = output.cpu()
        output_flat = output.view(-1, num_classes)
        loss = criterion(output_flat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += len(X) * loss.item()
    return total_loss / len(data.dataset)

def evaluate(model, criterion, data, device):
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in tqdm(data, "Evaluation"):
            X = X.permute((1,0,2)).to(device) # convert to seq first shape and move to desired device
            y = y.long()
            output = model(X)
            output = output.cpu()
            output_flat = output.view(-1, num_classes)
            y_true += y.tolist()
            y_pred += torch.argmax(output_flat, -1).tolist()
            total_loss += len(X) * criterion(output_flat, y).item()
    result = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
    result["loss"] = total_loss / len(data.dataset)
    result["pretty"] = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=False)
    return result


'''def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device) # (n, batch_size)

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target # (seq_len, batch_size) , (seq_len*batch_size,)'''

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"), init_token='<sos>', eos_token='<eos>', lower=True)
#train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
#TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 20
eval_batch_size = 10
dataset_path = ""
dataset_cache_name = "UrbanSound8k_cache.h5"
#train_data = batchify(train_txt, batch_size)
#val_data = batchify(val_txt, eval_batch_size)
#test_data = batchify(test_txt, eval_batch_size)

classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

num_classes = len(classes)
bptt = 35 # sequence len?

#ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
#model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

model = AudioTransformer(d_model=256, nhead=8, dim_feedforward=1024, num_layers=6, num_classes=num_classes, dropout=0.1)
criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

#Loop over epochs. Save the model if the validation loss is the best we’ve seen so far. Adjust the learning rate after each epoch.
best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None


if dataset_cache_name not in glob("*.h5"):
    dataset_filename = dataset.load(dataset_path, dataset_cache_name, False)

audio_data = h5py.File(dataset_cache_name, "r")
group = audio_data["urban_sound_8k"]

X_train = group["X_train"]
X_val = group["X_val"]
X_test = group["X_test"]
y_train = group["y_train"]
y_val = group["y_val"]
y_test = group["y_test"]

load_in_RAM = True
if load_in_RAM:
    X_train = X_train[:]
    X_val = X_val[:]
    X_test = X_test[:]
    y_train = y_train[:]
    y_val = y_val[:]
    y_test = y_test[:]
    audio_data.close()

train_dataloader = DataLoader(AudioDataset(X_train, y_train), batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(AudioDataset(X_val, y_val), batch_size, shuffle=False, drop_last=True)
test_dataloader = DataLoader(AudioDataset(X_test, y_test), batch_size, shuffle=False, drop_last=True)

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    print(train(model, criterion, optimizer, train_dataloader, device))
    val_loss = evaluate(model, criterion, val_dataloader, device)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

if not load_in_RAM:
    audio_data.close()