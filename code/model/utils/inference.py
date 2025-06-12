import torch
from typing import List, Dict, Optional, Union, Tuple, Literal
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from model.dataset.dataloader_functions import collate_func_pad

class InferenceEngine:
    """
    A flexible inference engine that handles both single-sample and ensemble prediction.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ensemble_strategy: Optional[Literal["majority", "logits_average", "confidence_weighted"]] = None,
    ):
        """
        Args:
            model: The trained model
            device: Device to run predictions on
            ensemble_strategy: If provided, uses this strategy for combining multiple predictions:
                - "majority": Most common prediction wins
                - "logits_average": Average logits then predict
                - "confidence_weighted": Weight votes by prediction confidence
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.ensemble_strategy = ensemble_strategy

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor], DataLoader, torch.utils.data.Dataset],
        return_logits: bool = False,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        return_confidence: bool = False,
        return_full_probs: bool = False,
        return_labels: bool = False,
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> Union[
        torch.Tensor,  # Just predictions
        Tuple[torch.Tensor, float],  # predictions, confidence
        Tuple[torch.Tensor, torch.Tensor],  # predictions, probabilities
        Tuple[torch.Tensor, torch.Tensor],  # predictions, labels
        Tuple[torch.Tensor, torch.Tensor, float],  # predictions, labels, confidence
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # predictions, labels, probabilities
        Dict[int, Union[torch.Tensor, Tuple[torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor], 
                       Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, float],
                       Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]  # Dataset results
    ]:
        """
        Run inference on input data. Handles single samples, multiple samples (ensemble), 
        or Dataset input.
        
        Args:
            inputs: Input data in one of four formats:
                - Single tensor [B, ...]
                - List of tensors for ensemble prediction
                - DataLoader yielding batches
                - Dataset to create DataLoader from
            return_logits: If True, return raw logits instead of predictions
            attention_mask: Optional attention mask(s) matching input format
            return_confidence: Whether to return confidence scores with predictions
            return_full_probs: Whether to return full probability distribution for all classes
            return_labels: Whether to return ground truth labels along with predictions
            batch_size: Batch size when creating DataLoader from Dataset (ignored for other input types)
            num_workers: Number of workers when creating DataLoader from Dataset (ignored for other input types)
            
        Returns:
            Predictions in a format matching the input:
            - Single tensor: predictions or (predictions, confidence/probs)
            - List of tensors: ensemble prediction or (prediction, confidence/probs)
            - Dataset/DataLoader: Dict mapping indices to predictions or (prediction, confidence/probs)
            If return_labels is True, labels are included in the return tuple for Dataset/DataLoader inputs
        """
        self.model.eval()
        
        # Case 1: Single tensor input
        if isinstance(inputs, torch.Tensor):
            if return_labels:
                raise ValueError("Cannot return labels for single tensor input")
            return self._predict_single(
                inputs, 
                attention_mask=attention_mask,
                return_logits=return_logits,
                return_confidence=return_confidence,
                return_full_probs=return_full_probs
            )
            
        # Case 2: List of tensors for ensemble
        elif isinstance(inputs, list) and isinstance(inputs[0], torch.Tensor):
            if return_labels:
                raise ValueError("Cannot return labels for list of tensors")
            if self.ensemble_strategy is None:
                raise ValueError("Must specify ensemble_strategy to use ensemble prediction")
            return self._predict_ensemble(
                inputs,
                attention_masks=attention_mask if isinstance(attention_mask, list) else None,
                return_logits=return_logits,
                return_confidence=return_confidence,
                return_full_probs=return_full_probs
            )
            
        # Case 3: Dataset input
        elif isinstance(inputs, torch.utils.data.Dataset):
            return self._predict_dataset(
                inputs,
                batch_size=1 if self.ensemble_strategy else batch_size,
                num_workers=num_workers,
                return_logits=return_logits,
                return_confidence=return_confidence,
                return_full_probs=return_full_probs,
                return_labels=return_labels
            )
            
        # Case 4: DataLoader input (maintained for backward compatibility)
        elif isinstance(inputs, DataLoader):
            return self._predict_dataset(
                inputs.dataset,
                batch_size=1 if self.ensemble_strategy else inputs.batch_size,
                num_workers=inputs.num_workers if hasattr(inputs, 'num_workers') else 0,
                return_logits=return_logits,
                return_confidence=return_confidence,
                return_full_probs=return_full_probs,
                return_labels=return_labels
            )
            
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

    def _predict_single(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        return_confidence: bool = False,
        return_full_probs: bool = False,
    ) -> Union[float, np.ndarray, Tuple[float, float], Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Handle prediction for a single input tensor."""
        inputs = inputs.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        output = self.model(inputs, attention_mask=attention_mask)
        
        if return_logits:
            return output.cpu().numpy()
            
        prediction = torch.argmax(output, dim=-1)
        
        if return_full_probs:
            probs = torch.softmax(output, dim=-1)
            return prediction.item(), probs.cpu().numpy()
            
        if return_confidence:
            probs = torch.softmax(output, dim=-1)
            confidence = torch.max(probs, dim=-1).values
            return prediction.item(), confidence.item()
            
        return prediction.item()

    def _predict_ensemble(
        self,
        samples: List[torch.Tensor],
        attention_masks: Optional[List[torch.Tensor]] = None,
        return_logits: bool = False,
        return_confidence: bool = False,
        return_full_probs: bool = False,
    ) -> Union[float, np.ndarray, Tuple[float, float], Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Handle ensemble prediction from multiple related samples.
        
        For each ensemble strategy, confidence and probabilities are calculated as follows:
        
        1. Majority Voting:
           - Confidence: Proportion of samples that agree with the final prediction
           - Full Probs: Normalized vote counts for each class (num_votes_for_class / total_votes)
        
        2. Logits Average:
           - Confidence: Softmax probability of the winning class from averaged logits
           - Full Probs: Full softmax distribution from averaged logits
        
        3. Confidence Weighted:
           - Confidence: Final probability of winning class after weighting individual predictions
           - Full Probs: Normalized weighted sum of each sample's probabilities, weighted by their
                        max probability (confidence). This gives more influence to samples where
                        the model is more certain.
        
        Args:
            samples: List of input tensors to ensemble
            attention_masks: Optional list of attention masks
            return_logits: If True, return raw stacked logits
            return_confidence: Whether to return confidence of winning class
            return_full_probs: Whether to return full probability distribution
            
        Returns:
            - If return_logits: stacked logits tensor
            - If return_full_probs: (predictions, probability distribution)
            - If return_confidence: (predictions, confidence score)
            - Otherwise: just predictions
        """
        all_logits = []
        
        for i, sample in enumerate(samples):
            mask = attention_masks[i] if attention_masks else None
            logits = self._predict_single(sample, mask, return_logits=True)
            all_logits.append(logits)
            
        stacked_logits = np.stack(all_logits)
        
        if return_logits:
            return stacked_logits
            
        if self.ensemble_strategy == "majority":
            # Get per-sample predictions
            predictions = np.argmax(stacked_logits, axis=-1)
            # Take most common prediction as final
            final_pred = float(np.bincount(predictions).argmax())
            
            if return_full_probs:
                # Count votes for each class
                vote_counts = np.zeros(stacked_logits.shape[-1])
                for pred in predictions:
                    vote_counts[pred] += 1
                # Convert to probabilities by normalizing
                probs = vote_counts / len(predictions)
                return final_pred, probs
                
            if return_confidence:
                # Confidence is proportion of samples that agree with final prediction
                confidence = float(np.mean(predictions == final_pred))
                return final_pred, confidence
                
        elif self.ensemble_strategy == "logits_average":
            # Average logits across samples
            avg_logits = np.mean(stacked_logits, axis=0)
            # Convert to probabilities
            probs = np.exp(avg_logits) / np.sum(np.exp(avg_logits), axis=-1, keepdims=True)
            # Get prediction from averaged logits
            final_pred = float(np.argmax(avg_logits))
            
            if return_full_probs:
                # Return full probability distribution
                return final_pred, probs
                
            if return_confidence:
                # Confidence is probability of winning class
                confidence = float(probs[final_pred])
                return final_pred, confidence
                
        elif self.ensemble_strategy == "confidence_weighted":
            # Get probabilities and confidences for each sample
            probs = np.exp(stacked_logits) / np.sum(np.exp(stacked_logits), axis=-1, keepdims=True)
            confidences = np.max(probs, axis=-1)  # [num_samples]
            
            # Weight each sample's probabilities by its confidence
            weighted_probs = np.sum(probs * confidences[:, np.newaxis], axis=0)
            # Normalize to get proper probability distribution
            weighted_probs = weighted_probs / np.sum(weighted_probs)
            
            final_pred = float(np.argmax(weighted_probs))
            
            if return_full_probs:
                # Return weighted probability distribution
                return final_pred, weighted_probs
                
            if return_confidence:
                # Confidence is weighted probability of winning class
                confidence = float(weighted_probs[final_pred])
                return final_pred, confidence
                
        return final_pred

    def _predict_batch(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        return_confidence: bool = False,
        return_full_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Handle prediction for a batch of inputs."""
        inputs = inputs.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        output = self.model(inputs, attention_mask=attention_mask)
        
        if return_logits:
            return output.cpu().numpy()
            
        prediction = torch.argmax(output, dim=-1)
        
        if return_full_probs:
            probs = torch.softmax(output, dim=-1)
            return prediction.cpu().numpy(), probs.cpu().numpy()
            
        if return_confidence:
            probs = torch.softmax(output, dim=-1)
            confidence = torch.max(probs, dim=-1).values
            return prediction.cpu().numpy(), confidence.cpu().numpy()
            
        return prediction.cpu().numpy()

    def _predict_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        num_workers: int = 0,
        return_logits: bool = False,
        return_confidence: bool = False,
        return_full_probs: bool = False,
        return_labels: bool = False,
    ) -> Union[
        Dict[int, Union[float, np.ndarray, Tuple[float, float], Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]],
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ]:
        """Handle prediction for a Dataset, optionally using ensemble prediction."""
        # Create appropriate DataLoader with collate function
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_func_pad
        )
        
        if self.ensemble_strategy:
            # Group samples by series/video ID
            series_samples: Dict[int, List[torch.Tensor]] = defaultdict(list)
            series_masks: Dict[int, List[torch.Tensor]] = defaultdict(list)
            series_labels: Dict[int, List[torch.Tensor]] = defaultdict(list)
            
            for batch_idx, batch in enumerate(dataloader):
                features, labels, attention_mask = batch
                # With batch_size=1, batch_idx equals sample_idx
                series_id = dataset.get_video_idx(batch_idx)
                
                series_samples[series_id].append(features)
                if attention_mask is not None:
                    series_masks[series_id].append(attention_mask)
                if return_labels:
                    series_labels[series_id].append(labels)
                    
            # Make predictions for each series
            predictions = {}
            for series_id in series_samples:
                masks = series_masks[series_id] if series_masks else None
                pred = self._predict_ensemble(
                    series_samples[series_id],
                    attention_masks=masks,
                    return_logits=return_logits,
                    return_confidence=return_confidence,
                    return_full_probs=return_full_probs
                )
                
                if return_labels:
                    # For ensemble, we'll use the label from the first sample
                    labels = series_labels[series_id][0].cpu().numpy()
                    if return_confidence:
                        predictions[series_id] = (pred[0], labels, pred[1])
                    elif return_full_probs:
                        predictions[series_id] = (pred[0], labels, pred[1])
                    else:
                        predictions[series_id] = (pred, labels)
                else:
                    predictions[series_id] = pred
                
        else:
            # Regular batch prediction
            all_predictions = []
            all_labels = []
            
            for batch in dataloader:
                features, labels, attention_mask = batch
                pred = self._predict_batch(
                    features,
                    attention_mask=attention_mask,
                    return_logits=return_logits,
                    return_confidence=return_confidence,
                    return_full_probs=return_full_probs
                )
                all_predictions.append(pred)
                if return_labels:
                    all_labels.append(labels)
            
            # Concatenate all predictions and labels
            if return_logits:
                return np.concatenate([p for p in all_predictions], axis=0)
            elif return_confidence or return_full_probs:
                # For (predictions, confidence/probs) tuples
                preds = np.concatenate([p[0] for p in all_predictions], axis=0)
                values = np.concatenate([p[1] for p in all_predictions], axis=0)
                if return_labels:
                    labels = torch.cat(all_labels, dim=0).cpu().numpy()
                    return preds, labels, values
                return preds, values
            else:
                preds = np.concatenate([p for p in all_predictions], axis=0)
                if return_labels:
                    labels = torch.cat(all_labels, dim=0).cpu().numpy()
                    return preds, labels
                return preds
                
        return predictions

    @torch.no_grad()
    def get_embeddings(
        self,
        inputs: Union[torch.Tensor, DataLoader],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get embeddings from the model's embedding layer.
        
        Args:
            inputs: Input tensor [B, ...] or DataLoader
            attention_mask: Optional attention mask for sequence data
            
        Returns:
            Embeddings tensor
        """
        if not hasattr(self.model, 'embedding') and not hasattr(self.model, 'embed'):
            raise AttributeError("Model does not have an embedding/embed method")
            
        embed_fn = getattr(self.model, 'embedding', None) or getattr(self.model, 'embed')
        
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            return embed_fn(inputs, attention_mask=attention_mask)
            
        all_embeddings = []
        for batch in inputs:
            if isinstance(batch, (tuple, list)):
                batch_inputs = batch[0].to(self.device)
                batch_mask = batch[2].to(self.device) if len(batch) > 2 else None
            else:
                batch_inputs = batch.to(self.device)
                batch_mask = None
                
            embeddings = embed_fn(batch_inputs, attention_mask=batch_mask)
            all_embeddings.append(embeddings)
            
        return torch.cat(all_embeddings, dim=0)

    def to(self, device: str) -> "InferenceEngine":
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        return self 