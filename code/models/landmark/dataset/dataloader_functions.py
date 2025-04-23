from torch.nn.utils.rnn import pad_sequence
import torch 

def collate_fn_pad(batch):
    features, labels = zip(*batch)  # Each feature is [T, D]
    
    lengths = [x.shape[0] for x in features]
    padded_features = pad_sequence(features, batch_first=True)  # Shape: [B, T_max, D]
    
    attention_masks = torch.zeros_like(padded_features[:, :, 0])
    for i, l in enumerate(lengths):
        attention_masks[i, :l] = 1
    
    return padded_features, torch.tensor(labels), attention_masks