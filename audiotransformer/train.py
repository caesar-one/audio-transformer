from transformers import AdamW, get_linear_schedule_with_warmup

IN_COLAB = True
try:
    from google.colab import drive
except:
    IN_COLAB = False
if IN_COLAB:
    drive.mount('/content/drive')
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import time
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from audiotransformer.dataset import AudioDataset
from audiotransformer import dataset
from models.transformers import AudioTransformer
from sklearn.metrics import classification_report
from glob import glob
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    print("Cuda device:", torch.cuda.get_device_name())

learning_rate = 5e-5
weight_decay = 0.0
adam_epsilon = 1e-8

max_grad_norm = 1.0
num_train_epochs = 3


# Trains a single epoch with hyper-parameters provided
def train(model, criterion, optimizer, data, device):
    model.train()  # Turn on the train mode
    total_loss = 0.
    progress = tqdm(data, "Training")
    for X, y in progress:
        X = X.to(device)
        y = y.long()
        optimizer.zero_grad()
        output = model(X)
        output = output.cpu()
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += len(X) * loss.item()
    return total_loss / len(data.dataset)


def evaluate(model, criterion, data, device):
    model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in tqdm(data, "Evaluation"):
            X = X.to(device)
            y = y.long()
            output = model(X)
            output = output.cpu()
            y_true += y.tolist()
            y_pred += torch.argmax(output, -1).tolist()
            total_loss += len(X) * criterion(output, y).item()
    result = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
    result["loss"] = total_loss / len(data.dataset)
    result["pretty"] = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=False)
    return result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 20
eval_batch_size = 10
dataset_path = ""
dataset_cache_name = "UrbanSound8k_cache.h5"
drive_path = "/content/drive/My Drive/"

classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

num_classes = len(classes)

model = AudioTransformer(d_model=256, nhead=8, dim_feedforward=1024, num_layers=6, num_classes=num_classes, dropout=0.1)

best_val_loss = float("inf")
epochs = 25  # The number of epochs
best_model = None

# Dataset load using caching system
if IN_COLAB:
    if dataset_cache_name not in glob("*.h5"):
        print("No dataset cache in runtime, ", end="")
        if drive_path + dataset_cache_name not in glob(drive_path + "*.h5"):
            print("creating... ", end="")
            dataset_filename = dataset.load(dataset_path, dataset_cache_name, False)
            #!cp {dataset_cache_name} / content / drive / My\ Drive /
        else:
            print("copying from Drive... ", end="")
            #!cp / content / drive / My\ Drive / {dataset_cache_name}.
        print("Done!")
else:
    if dataset_cache_name not in glob("*.h5"):
        print("No dataset cache in runtime, creating... ")
        dataset_filename = dataset.load(dataset_path, dataset_cache_name, False)
        print("Done!")

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

warmup_steps = len(train_dataloader)
num_training_steps = t_total = int(len(train_dataloader) * num_train_epochs)
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
#scheduler2 = get_cosine_with_hard_restarts_schedule_with_warmup(
#            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps, num_cycles=3.0
#        )
criterion = nn.CrossEntropyLoss()

# Initialize Tensorboard and writer
#if IN_COLAB:
#    %tensorboard --logdir runs
tensorboard_writer = SummaryWriter(log_dir="runs")
tensorboard_writer.add_graph(model, input_to_model=torch.zeros((16,174,256)))

model.to(device)
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    # Train
    train_loss = train(model, criterion, optimizer, train_dataloader, device)
    print("Train Loss:", train_loss)
    # Evaluate
    val_results = evaluate(model, criterion, val_dataloader, device)
    print("Val Loss:", val_results["loss"])
    # Plot results on Tensorboard
    tensorboard_writer.add_scalars('Metrics', {'val accuracy': val_results["accuracy"],
                                               'val f1-score': val_results["macro avg"]["f1-score"]}, epoch)
    tensorboard_writer.add_scalars('Loss', {'train': train_loss, 'val': val_results["loss"]}, epoch)
    tensorboard_writer.add_text("Val report",
                                "Val Loss: " + str(val_results["loss"]) + "\nClassification report:\n" + str(val_results[
                                    "pretty"]), epoch)
    # print("Val Loss:",val_results["loss"],"\nClassification report:\n",val_results["pretty"])

if not load_in_RAM:
    audio_data.close()