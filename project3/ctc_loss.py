import torch
import numpy as np
from math import inf

class CTCLoss(torch.nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()

    def forward(self, prediction, target, pred_lens, target_lens):
        """
        Forward alpha probabilities calculated using dynamic programming.

        prediction: Tensor of probabilities at each step given input at each step. 
                    Seq_length * Batch_size * Alphabet_size, padded.
        target: Tensor indicating the sequence to map input to. 
                Batch_size * Word_length, padded.
        pred_lens: Tensor indicating length of input, without padding.
        target_lens: Tensor indicating length of output, without padding.
        """

        # Create an alpha for each timestep and populate with neural net predictions for each letter + blank
        num_states = 2 * max(target_lens) + 1
        num_steps = max(pred_lens)
        batch_size = prediction.size()[1]
        # Input_dim * Batch_dim * Alphabet_dim
        # Attempting to do everything in parallel, where dim 1 tracks different trellis for different batch elements
        alphas = torch.zeros((num_steps, batch_size, num_states))

        # For each batch, get tensor list of letter indices
        # One list of letter indices for each batch element
        w_idx = target.tolist()

        # Set all alphas to be the neural net output probabilities at each time step for each batch
        # Index prob of blank at each timestep for each batch element and repeat it the same number of times as blank nodes
        # NOTE: this implementation assume a blank node before and after the silence nodes
        blank_probs = prediction[:num_steps,:,0].repeat(max(target_lens)+1,1,1)
        alphas[:,:,range(0,num_states,2)] = blank_probs.view(num_steps,batch_size,-1)

        # Index probs of each letter of each target word for each batch and at each timestep
        b_idx = torch.arange(batch_size).view(-1,1)
        alphas[:,:,range(1,num_states-1,2)] = prediction[:num_steps,b_idx,w_idx]

        # For each time step
        for tid in range(num_steps):
            if tid == 0:
                # Set initial states to be the silence and blank node in the first timestep
                # All other probs zero 
                alphas[tid,:,:] = torch.log(torch.tensor(0.0001))
                alphas[tid,:,0] = torch.log(torch.tensor(0.499))
                alphas[tid,:,1] = torch.log(torch.tensor(0.499))
                continue
            # Get previous alphas
            a = alphas[tid-1,:,:]
            # Set alpha of first silence state to the same as previous
            alphas[tid,:,0] += a[:,0]
            # Set alpha of the second state as sum of first two alphas times the output probability from LSTM
            d = a[:,0:2].clone()
            alphas[tid,:,1] += torch.logsumexp(d, dim=1)
            a = a[:,1:]
            # to prevent backward gradient tracking issues
            b = a.unfold(1, 3, 2).clone()
            c = a.unfold(1, 2, 2).clone()
            # For each letter state, logsumexp the alphas of the previous letter state, 
            # blank state above, and letter state above that
            alphas[tid,:,range(3,num_states-1,2)] += torch.logsumexp(b, dim=2)
            # For each blank state, logsumexp the alphas of the previous blank state and letter state above
            alphas[tid,:,range(2,num_states,2)] += torch.logsumexp(c, dim=2)

        # Get final alphas across each batch, using given indices of input 
        # sequence length and target word length
        log_marginal_prob = torch.tensor(0.)

        for b in range(batch_size):
            sid = 2 * target_lens[b]
            tid = pred_lens[b] - 1
            # The final states are the last two states, since it could be either blank or silence
            final_states = torch.zeros((2))
            final_states[0] = alphas[tid,b,sid-1]
            final_states[1] = alphas[tid,b,sid]
            # Sum final two state probabilities
            log_marginal_prob+=torch.logsumexp(final_states, dim=0)

        # apply mean reduction over batches
        log_marginal_prob/=batch_size
        
        # output positive log prob to make it a min optimization
        return -1.*log_marginal_prob