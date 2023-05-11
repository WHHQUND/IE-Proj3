import pdb

import torch
import librosa
import os
import numpy as np
import string
import pandas as pd
from torch.utils.data import Dataset

# I would separate the discrete and MFCC to two fuctions
class AsrDataset_Discrete(Dataset):

    def __init__(self, file_lbls, lbl_names='data/clsp.lblnames', text=None):

        # assert self.feature_type in ['discrete', 'mfcc']

        # self.blank = "<blank>"
        # self.silence = "<sil>"

        # === write your code here ===
        
        # Create a dictionary which store the alphabet
        phones = {'_':27,
                'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7,
                'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 'p':16,
                'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22,
                'w':23, 'x':24, 'y':25, 'z':26}
        phones_inv = {v: k for k, v in phones.items()}
        self.phones = phones
        self.phones_inv = phones_inv
        self.text = text

        # Create vocab and label to index
        lblnames = []
        with open(lbl_names, 'r') as n:
            lines = n.readlines()[1:]
            for l in lines:
                lblnames.append(l.strip('\n'))
        index = [num for num in range(1,len(lblnames)+1)]
        self.vocab = {lblnames[i] : index[i] for i in range(len(lblnames))}

        # reate word_labels, used in train_test_split
        self.word_labels = []
        self.dataset = self.load_quantized_features(file_lbls, text=text)
        self.word_labels = np.array(self.word_labels)

    def load_quantized_features(self, file_lbls, text=None):
        dataset = {}

        # Extract labels for each utterance and convert to index tensor
        lbl_seqs = []
        with open(file_lbls, 'r') as t:
            lines = t.readlines()[1:]
            for j in range(len(lines)):
                lbls = lines[j].split(" ")[:-1]
                l_tensor = []
                for lbl in lbls:
                    l_idx = self.vocab.get(lbl)
                    l_tensor.append(l_idx)
                lbl_seqs.append(torch.tensor(l_tensor))

        # Read in the words and convert to index tensor
        if text is not None:
            words = []
            with open(text, 'r') as s:
                lines = s.readlines()[1:]
                for l in lines:
                    w = '_' + l.strip('\n') + '_'
                    w_tensor = []
                    for let in list(w):
                        w_idx = self.phones.get(let)
                        w_tensor.append(w_idx)
                    words.append(torch.tensor(w_tensor))

            for idx in range(len(lbl_seqs)):
                dataset.update({idx: {'features':lbl_seqs[idx], 'target_tokens':words[idx]}})
                w_lbl = int(''.join(map(str, words[idx].detach().numpy().tolist())))
                self.word_labels.append(w_lbl)
        else: 
            for idx in range(len(lbl_seqs)):
                dataset.update({idx: {'features':lbl_seqs[idx]}})

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.text is None:
            return self.dataset[idx]['features'], None
        return self.dataset[idx]['features'], self.dataset[idx]['target_tokens']


    
    
class AsrDataset_MFCC(Dataset):
    def __init__(self, wav_scp, wav_dir, text=None):
        
        phones = {'_':27,
                    'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7,
                    'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15, 'p':16,
                    'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22,
                    'w':23, 'x':24, 'y':25, 'z':26}
        phones_inv = {v: k for k, v in phones.items()}
        self.phones = phones
        self.phones_inv = phones_inv
        self.text = text
        self.word_labels = []
        self.dataset = self.compute_mfcc(wav_scp, wav_dir, text=text)
        self.word_labels = np.array(self.word_labels)

    def compute_mfcc(self, wav_scp, wav_dir, text=None):
        dataset = {}

        # Create my own function to compute MFCC
        mfccs = []
        with open(wav_scp, 'r') as t:
            paths = t.readlines()[1:]
            for j in range(len(paths)):
                path = wav_dir + '/' + paths[j].strip('\n')
                audio, sr = librosa.load(path, sr=None)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=100, n_fft=int(0.050*sr), hop_length=int(0.020*sr))
                mfccs.append(torch.swapdims(torch.tensor(mfcc), 0, 1))

        # Read in the words
        if text is not None:
            words = []
            with open(text, 'r') as s:
                lines = s.readlines()[1:]
                for i in lines:
                    w = '_' + i.strip('\n') + '_'
                    w_tensor = []
                    for l in list(w):
                        w_idx = self.phones.get(l)
                        w_tensor.append(w_idx)
                    words.append(torch.tensor(w_tensor))

            for idx in range(len(mfccs)):
                dataset.update({idx: {'features':mfccs[idx], 'target_tokens':words[idx]}})
                w_lbl = int(''.join(map(str, words[idx].detach().numpy().tolist())))
                self.word_labels.append(w_lbl)
        else:
            for idx in range(len(mfccs)):
                dataset.update({idx: {'features':mfccs[idx]}})

        return dataset
    
    # This function is provided
    # def compute_mfcc(self, wav_scp, wav_dir):
    #     """
    #     Compute MFCC acoustic features (dim=40) for each wav file.
    #     :param wav_scp:
    #     :param wav_dir:
    #     :return: features: List[np.ndarray, ...]
    #     """
    #     features = []
    #     with open(wav_scp, 'r') as f:
    #         for wavfile in f:
    #             wavfile = wavfile.strip()
    #             if wavfile == 'jhucsp.trnwav':  # skip header
    #                 continue
    #             wav, sr = librosa.load(os.path.join(wav_dir, wavfile), sr=None)
    #             feats = librosa.feature.mfcc(y=wav, sr=16e3, n_mfcc=40, hop_length=160, win_length=400).transpose()
    #             features.append(feats)
    #     return features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.text is None:
            return self.dataset[idx]['features'], None
        return self.dataset[idx]['features'], self.dataset[idx]['target_tokens']
