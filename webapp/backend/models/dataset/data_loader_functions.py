import torch
from typing import List, Tuple

def collate_func_pad(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Collate function for padding variable-length sequences into a batch.
    Each item in the batch is expected to be a tuple of (features, labels, attention_mask).

    Args:
        batch: List of tuples, where each tuple contains:
            - features: torch.Tensor of shape (seq_len, feature_dim)
            - labels: torch.Tensor of shape (1,) or (seq_len,)
            - attention_mask: torch.Tensor of shape (seq_len,)

    Returns:
        Tuple of:
            - padded_features: Tensor of shape (batch_size, max_seq_len, feature_dim)
            - padded_labels: Tensor of shape (batch_size, max_seq_len)
            - padded_attention_mask: Tensor of shape (batch_size, max_seq_len)
    """
    features, labels, attention_masks = zip(*batch)

    seq_lengths = [f.shape[0] for f in features]
    max_len = max(seq_lengths)

    feature_dim = features[0].shape[1]

    # Initialize padded tensors
    padded_features = torch.zeros((len(batch), max_len, feature_dim))
    padded_labels = torch.zeros((len(batch), max_len), dtype=torch.long)
    padded_attention_masks = torch.zeros((len(batch), max_len))

    for i, (f, l, m) in enumerate(zip(features, labels, attention_masks)):
        length = f.shape[0]
        padded_features[i, :length, :] = f
        padded_labels[i, :length] = l if l.ndim > 0 else l.expand(length)
        padded_attention_masks[i, :length] = m

    return padded_features, padded_labels, padded_attention_masks
