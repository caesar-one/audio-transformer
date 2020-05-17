import h5py
import librosa
import pandas as pd
from glob import glob
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch

DATASET_PATH = "../UrbanSound8K/audio/"
METADATA_PATH = "../UrbanSound8K/metadata/UrbanSound8K.csv"

class AudioDataset(Dataset):
    def __init__(self, X_desc, y_desc):
        """
        A PyTorch iterable dataset, suitable for both HDF5 file descriptors or NumPy arrays.

        :param X_desc: HDF5 file descriptor or NumPy array for the features
        :param y_desc: HDF5 file descriptor or NumPy array for the labels
        :return: An iterator that can be used with the torch.utils.data.DataLoader class.
        """
        self.X_desc = X_desc
        self.y_desc = y_desc

    def __len__(self):
        return self.y_desc.shape[0]

    def __getitem__(self, idx):
        return (self.X_desc[idx], self.y_desc[idx])

'''def get_proper_shape(batch): # shape (batch, spec_row, spec_col)
    print(type(batch))
    print(len(batch), batch[0][0].shape)
    batch = torch.from_numpy(batch).permute(2,0,1) #todo check proper output shaping
    print(batch.shape,"\n")
    #assert batch.shape == (173, 20, 256)
    #batch = torch.cat([torch.zeros((1, 20, 256)), batch], dim = 0)
    assert batch.shape == (174, 20, 256)
    return batch.contiguous() # shape (spec_col, batch, spec_row)'''


# The main loading function.
def load(path: str = "", save_filename: str = "audio_data.h5", debug: bool = False) -> str:
    """
    Loads the audio clips and generates training, testing and evaluating data from them.
    Each set of data is a np.ndarray shape (# of clips, T, # of channels, y_size, x_size). See later for details.

    :param path: absolute path of the main folder of the audio clips
    :param save_filename: the name of the file which will contain the generated data
        Otherwise, the split has T contiguous slots.
        Each slot has length *samples_num* / *num_frames*
    :param debug: If True, load just a small part of the data in order to debug
    :return: the path at which is stored the dataset.
    """

    audio_data = h5py.File(path + save_filename, "w-")
    group_name = "urban_sound_8k"
    audio_data.create_group(group_name)
    dataset = audio_data[group_name]

    # DATASET CREATION
    # We split the data into 3 sets: train (~60%), val (~20%), test (~20%).

    # Assign folders to the appropriate set
    wav_paths = glob(path + DATASET_PATH + "**/*.wav", recursive=True)
    wav_paths_train, wav_paths_val, wav_paths_test = [], [], []
    for p in wav_paths:
        if p.split("/")[-2] in ["fold1", "fold2"]:
            wav_paths_test.append(p)
        elif p.split("/")[-2] in ["fold3", "fold4"]:
            wav_paths_val.append(p)
        else:
            wav_paths_train.append(p)

    # Load the metadata
    metadata = pd.read_csv(path + METADATA_PATH)
    # Create a mapping from audio clip names to their respective label IDs
    name2class = dict(zip(metadata["slice_file_name"], metadata["classID"]))

    if debug:
        wav_paths_test, wav_paths_train, wav_paths_val = wav_paths_test[:64], wav_paths_train[:64], wav_paths_val[:64]
    X_train = dataset.create_dataset("X_train", shape=(len(wav_paths_train),1+173,256)) # we add an extra dimension for a special [cls] token
    y_train = dataset.create_dataset("y_train", shape=(len(wav_paths_train),))
    X_val = dataset.create_dataset("X_val", shape=(len(wav_paths_val),1+173,256))
    y_val = dataset.create_dataset("y_val", shape=(len(wav_paths_val),))
    X_test = dataset.create_dataset("X_test", shape=(len(wav_paths_test),1+173,256))
    y_test = dataset.create_dataset("y_test", shape=(len(wav_paths_test),))

    #_min, _max = np.float('inf'), np.float('-inf')

    for paths, setname in zip([wav_paths_train, wav_paths_val, wav_paths_test], ["train", "val", "test"]):
        counter = 0
        for wav_path in tqdm(paths, desc=f"Converting {setname} samples in spectrograms"):
            # Load the audio clip stored at *wav_path* in an audio array
            audio_array, sr = librosa.load(wav_path)
            samples_num = sr * 4
            # Truncate/pad arrays so that they all have the same size
            audio_array = audio_array[:samples_num]
            reshaped_array = np.zeros((samples_num,))
            reshaped_array[:audio_array.shape[0]] = audio_array
            #reshaped_array[0] += 0.5
            # Create spectrogram
            spec = librosa.feature.melspectrogram(reshaped_array, n_mels=256, hop_length=512) # todo add correct vals
            spec = librosa.power_to_db(spec)
            # add an extra special time instant as [cls] token for transformer classification with 0.5 value
            # to let it be different from the padding
            spec = np.concatenate([np.zeros((256, 1)) + 0.5, spec], axis=1)

            # Note: spec is transposed in order to have it time-instant major (as each time instant is a sequence token)
            spec = spec.transpose((1,0))
            assert spec.shape == (174, 256)
            # Append each frames list to their respective set
            audio_filename = wav_path.split("/")[-1]
            if setname == "train":
                X_train[counter] = spec
                y_train[counter] = int(name2class[audio_filename])
            elif setname == "val":
                X_val[counter] = spec
                y_val[counter] = int(name2class[audio_filename])
            else:
                X_test[counter] = spec
                y_test[counter] = int(name2class[audio_filename])

            #_min, _max = np.min([_min, np.min(spec)]), np.max([_max, np.max(spec)])

            counter += 1
    audio_data.close()
    return path + save_filename