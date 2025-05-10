import torch.nn as nn
import torch
from typing import Optional, Tuple


class RNNClassifier(nn.Module):
    """
    GRU-based classifier for sequence classification.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state
        num_layers (int): Number of GRU layers
        num_classes (int): Number of output classes
    """
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, num_classes: int
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def embedding(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the sequence embedding from the last layer's hidden state.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            attention_mask: Optional mask tensor of shape [batch_size, seq_len]
            
        Returns:
            Tensor of shape [batch_size, hidden_size]
        """
        if attention_mask is not None:
            # Get sequence lengths from attention mask
            lengths = attention_mask.sum(dim=1).cpu()
            # Pack the padded sequence
            packed_x = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            # Run GRU
            packed_output, hidden = self.rnn(packed_x)
            return hidden[-1]  # Return last layer's hidden state
        else:
            output, hidden = self.rnn(x)
            return hidden[-1]

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size (B), seq_len (T), input_size (D)]
            attention_mask: Optional mask tensor of shape [batch_size (B), seq_len (T)]
            
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        if attention_mask is not None:
            # Get sequence lengths from attention mask
            lengths = attention_mask.sum(dim=1).cpu()
            # Pack the padded sequence
            packed_x = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            # Run GRU
            packed_output, hidden = self.rnn(packed_x)
            # Use last layer's hidden state
            out = self.fc(hidden[-1])
            return out
        else:
            output, hidden = self.rnn(x)
            out = self.fc(hidden[-1])
            return out


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sequence classification.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of output classes
    """
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, num_classes: int
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def embedding(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the sequence embedding from the last layer's hidden state.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            attention_mask: Optional mask tensor of shape [batch_size, seq_len]
            
        Returns:
            Tensor of shape [batch_size, hidden_size]
        """
        if attention_mask is not None:
            # Get sequence lengths from attention mask
            lengths = attention_mask.sum(dim=1).cpu()
            # Pack the padded sequence
            packed_x = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            # Run LSTM
            packed_output, (hidden, cell) = self.lstm(packed_x)
            return hidden[-1]  # Return last layer's hidden state
        else:
            output, (hidden, cell) = self.lstm(x)
            return hidden[-1]

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size (B), seq_len (T), input_size (D)]
            attention_mask: Optional mask tensor of shape [batch_size (B), seq_len (T)]
            
        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        if attention_mask is not None:
            # Get sequence lengths from attention mask
            lengths = attention_mask.sum(dim=1).cpu()
            # Pack the padded sequence
            packed_x = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            # Run LSTM
            packed_output, (hidden, cell) = self.lstm(packed_x)
            # Use last layer's hidden state
            out = self.fc(hidden[-1])
            return out
        else:
            output, (hidden, cell) = self.lstm(x)
            out = self.fc(hidden[-1])
            return out
