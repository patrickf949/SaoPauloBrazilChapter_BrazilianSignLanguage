from torch.nn.utils.rnn import pad_sequence
import torch

verbose = True


def collate_func_pad(batch):
    features, labels = zip(*batch)  # Each feature is [T, D]

    lengths = [x.shape[0] for x in features]
    padded_features = pad_sequence(features, batch_first=True)  # Shape: [B, T_max, D]

    # Debug information about padding
    if verbose:
        if len(set(lengths)) > 1:
            print(f"Padding was needed. Sequence lengths in batch: {lengths}")
            print(f"Padded to length: {padded_features.shape[1]}")

    attention_masks = torch.zeros_like(padded_features[:, :, 0])
    for i, length in enumerate(lengths):
        attention_masks[i, :length] = 1

    return padded_features, torch.tensor(labels), attention_masks
