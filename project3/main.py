import pdb

import argparse
import numpy as np
import torch

from dataset import AsrDataset_MFCC, AsrDataset_Discrete
from model import LSTM_ASR_MFCC, LSTM_ASR_Discrete
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from ctc_loss import CTCLoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.special import softmax

def collate_fn(batch):
    data = [i[0] for i in batch]
    targets = [i[1] for i in batch]
    data_lens = [len(x) for x in data]
    target_lens = [len(x) for x in targets]
    data_pad = pad_sequence(data, batch_first=True, padding_value=0)
    target_pad = pad_sequence(targets, batch_first=True, padding_value=0)
    return {'features':data_pad, 'targets':target_pad, 'feat_lens':data_lens, 'target_lens':target_lens}


def get_vocab(file_name, func=lambda x: x, skip_header=True):
    res = list()
    with open(file_name, "r") as fin:
        if skip_header:
            fin.readline()  # skip the header
        for l in fin:
            if len(l.strip()) == 0:
                continue
            fields = func(l.strip())
            res.append(fields)
    res = list(set(res))
    return res


def train(train_dataloader, val_dataloader, model, criterion, optim):
    model.train()
    total_loss = 0.
    num_batches = 0
    for i, item in enumerate(train_dataloader):
        input_seq = item['features']
        target_seq = item['targets']
        input_lens = item['feat_lens']
        target_lens = item['target_lens']
        model.zero_grad()
        pred_seq = model(input_seq, input_lens)
        
        # compute loss and back propagate
        loss = criterion(torch.swapdims(pred_seq, 0, 1), target_seq, input_lens, target_lens)
        total_loss+=loss.item()
        loss.backward()
        
        # optimize
        optim.step()

        num_batches+=1

    total_loss/=num_batches
    

    model.eval()
    val_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for i, item in enumerate(val_dataloader):
            input_seq_val = item['features']
            target_seq_val = item['targets']
            input_lens_val = item['feat_lens']
            target_lens_val = item['target_lens']

            pred_seq_val = model(input_seq_val, input_lens_val)

            loss = criterion(torch.swapdims(pred_seq_val, 0, 1), target_seq_val, input_lens_val, target_lens_val)
            val_loss+=loss.item()
            num_batches+=1

    val_loss/=num_batches

    return total_loss, val_loss


def test(test_dataset, phones, phones_inv, vocab, model, criterion, not_test = True):

    
    model.eval()
    correct = 0
    total = 0
    string = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            input_seq, target_seq = test_dataset[i]

            # add batch dim
            input_seq = torch.unsqueeze(input_seq, dim=0)
            
            if not_test:
                # turn into string
                target_word = ''
                for widx in target_seq.tolist():
                    target_word+=phones_inv.get(widx)

            input_len = [int(input_seq.size()[1])]
            pred_seq = model(input_seq, input_len)

            losses = []
            for word in vocab:
                # Turn word into a tensor of letter indices
                w_tensor = []
                word = '_' + word + '_'
                for let in list(word):
                    w_idx = phones.get(let)
                    w_tensor.append(w_idx)
                target = torch.unsqueeze(torch.tensor(w_tensor), dim=0)

                input_len = (pred_seq.size()[1],)
                target_len = (target.size()[1],)

                loss = criterion(torch.swapdims(pred_seq, 0, 1), target, input_len, target_len)
                losses.append(loss.item())

            losses = -1. * np.array(losses)
            pred_word = vocab[np.argmax(losses)]
            confidence = np.amax(softmax(losses))
            string.append((pred_word, confidence))
            if not_test:
                
                pred_word = '_' + pred_word + '_'
                if target_word == pred_word:
                    correct+=1
            total+=1
    
    return string, correct / total


if __name__ == "__main__":
    vocab = get_vocab("data/clsp.trnscr")
    model = None

    criterion = torch.nn.CTCLoss()
    train_dataset = AsrDataset_Discrete('data/clsp.trnlbls', text='data/clsp.trnscr')
    train_idx, val_idx = train_test_split(np.arange(len(train_dataset)), test_size=0.2, shuffle=True, stratify=train_dataset.word_labels)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    val_dataset = []
    for vid in val_idx.tolist():
        val_dataset.append(train_dataset[vid])
    train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, sampler=train_sampler)
    val_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, sampler=val_sampler)
    test_dataset = AsrDataset_Discrete('data/clsp.devlbls')
    model = LSTM_ASR_Discrete(embedding_dim=300, hidden_dim=200)
        
    print('Training Discrete Model')
    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    train_losses = []
    val_losses = []
    for epoch in range(100):
        train_loss, val_loss = train(train_dataloader, val_dataloader, model, criterion, optim)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('\tepoch ', str(epoch), ': train_loss=', str(train_loss), '  |  val_loss=', str(val_loss))


    # Plot train and val loss
    fig0=plt.figure(0)
    plt.plot(train_losses, color='red', label='train')
    plt.xlabel('Epoch number')
    plt.ylabel('Discrete Epoch loss')
    plt.plot(val_losses, color='blue', label='val')
    plt.legend()
    plt.show()
    
    print('Test Discrete Model')
    model.load_state_dict(torch.load('checkpoint/model_best_primary.pt'))
    predictions, accuracy = test(train_dataset, train_dataset.phones, train_dataset.phones_inv, vocab, model, criterion)
    print('Final Train Accuracy: ', str(accuracy))
    print('Train finished')
    predictions, accuracy = test(val_dataset, train_dataset.phones, train_dataset.phones_inv, vocab, model, criterion)
    print('Final Validation Accuracy: ', str(accuracy))
    print('Val finished')
    predictions, accuracy = test(test_dataset, train_dataset.phones, train_dataset.phones_inv, vocab, model, criterion, not_test=False)
    # print('Final Test Accuracy: ' + str(accuracy))

    file = open("Discrete_test_results.txt", "w")
    file.write("test_results.txt\n")
    file.write("predicted_word\tconfidence\n")
    for most_likely_word, confidence in predictions:
        file.write(f'{most_likely_word}\t{confidence}\n')
    file.close()
        
        
        
        
    train_dataset_MFCC = AsrDataset_MFCC('data/clsp.trnwav', 'data/waveforms', text='data/clsp.trnscr')
    train_idx_MFCC, val_idx_MFCC = train_test_split(np.arange(len(train_dataset_MFCC)), test_size=0.2, shuffle=True, stratify=train_dataset_MFCC.word_labels)
    train_sampler_MFCC = SubsetRandomSampler(train_idx_MFCC)
    val_sampler_MFCC = SubsetRandomSampler(val_idx_MFCC)
    val_dataset_MFCC = []
    for vid in val_idx_MFCC.tolist():
        val_dataset_MFCC.append(train_dataset_MFCC[vid])
    train_dataloader_MFCC = DataLoader(train_dataset_MFCC, batch_size=128, collate_fn=collate_fn, sampler=train_sampler_MFCC)
    val_dataloader_MFCC = DataLoader(train_dataset_MFCC, batch_size=128, collate_fn=collate_fn, sampler=val_sampler_MFCC)
    test_dataset_MFCC = AsrDataset_MFCC('data/clsp.devwav', 'data/waveforms')
    model_MFCC = LSTM_ASR_MFCC(mfcc_dim=100, hidden_dim=200)
        
        
    print('Training MFCC Model')
    optim = torch.optim.Adam(model_MFCC.parameters(), lr=0.005)
    train_losses = []
    val_losses = []
    for epoch in range(100):
        train_loss, val_loss = train(train_dataloader_MFCC, val_dataloader_MFCC, model_MFCC, criterion, optim)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('\tepoch ', str(epoch), ': train_loss=', str(train_loss), '  |  val_loss=', str(val_loss))


    # Plot train and val loss
    fig0=plt.figure(0)
    plt.plot(train_losses, color='red', label='train')
    plt.xlabel('Epoch number')
    plt.ylabel('MFCC Epoch loss')
    plt.plot(val_losses, color='blue', label='val')
    plt.legend()
    plt.show()
        
        
    print('Test MFCC Model')
    predictions, accuracy = test(train_dataset_MFCC, train_dataset_MFCC.phones, train_dataset_MFCC.phones_inv, vocab, model_MFCC, criterion)
    print('Final Train Accuracy: ', str(accuracy))
    print('Train finished')
    predictions, accuracy = test(val_dataset_MFCC, train_dataset_MFCC.phones, train_dataset_MFCC.phones_inv, vocab, model_MFCC, criterion)
    print('Final Validation Accuracy: ', str(accuracy))
    print('Val finished')
    predictions, accuracy = test(test_dataset_MFCC, train_dataset_MFCC.phones, train_dataset_MFCC.phones_inv, vocab, model_MFCC, criterion, not_test=False)
    # print('Final Test Accuracy: ' + str(accuracy))

    file = open("MFCC_test_results.txt", "w")
    file.write("test_results.txt\n")
    file.write("predicted_word\tconfidence\n")
    for most_likely_word, confidence in predictions:
        file.write(f'{most_likely_word}\t{confidence}\n')
    file.close()
    
    
