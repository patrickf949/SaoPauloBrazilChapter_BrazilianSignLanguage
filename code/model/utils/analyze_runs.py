import json
import os
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Dict, Tuple
from omegaconf import OmegaConf, DictConfig

from model.dataset.landmark_dataset import LandmarkDataset
from model.utils.evaluate import EvaluationMetrics
from model.utils.inference import InferenceEngine
from model.utils.path_utils import load_base_paths
from model.utils.utils import load_obj

def get_results_from_training_log(training_log):
    total_epochs = len(training_log)
    max_epoch = training_log['epoch'].max()
    if max_epoch != total_epochs:
        print(f"Warning: Training log has {total_epochs} epochs, but the last epoch is {max_epoch}")

    final_epoch = training_log['epoch'].iloc[-1]
    final_train_loss = training_log['avg_train_loss'].iloc[-1]
    final_val_loss = training_log['avg_val_loss'].iloc[-1]

    # Get best model
    best_model_idx = training_log['avg_val_loss'].idxmin()
    best_epoch = training_log.loc[best_model_idx, 'epoch']
    best_train_loss = training_log.loc[best_model_idx, 'avg_train_loss']
    best_val_loss = training_log.loc[best_model_idx, 'avg_val_loss']

    results = {
        'total_epochs': total_epochs,
        'best_epoch': best_epoch,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'final_epoch': final_epoch,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
    }

    return results

def get_results_from_config(config):
    model = config['model']['class_name'].split('.')[-1]
    position = 'position' in config['features'].keys()
    angles = 'angles' in config['features'].keys()
    differences = 'differences' in config['features'].keys()
    distances = 'distances' in config['features'].keys()
    metadata = 'metadata' in config['features'].keys()
    p_rotate = config['augmentation']['train']['rotate']['p']
    p_noise = config['augmentation']['train']['noise']['p']

    results = {
        'model': model,
        'position': position,
        'angles': angles,
        'differences': differences,
        'distances': distances,
        'metadata': metadata,
        'p_rotate': p_rotate,
        'p_noise': p_noise,
    }
    return results

def make_results_row(run_dir, run_name):
    results = {
        'run_name': run_name,
    }
    training_log = pd.read_csv(os.path.join(run_dir, run_name, "training_log.csv"))
    config = OmegaConf.load(os.path.join(run_dir, run_name, "config.yaml"))
    results.update(get_results_from_training_log(training_log))
    results.update(get_results_from_config(config))
    return results

def load_model(model_dir: str, model_name: str = "best_model.pt", device: str = "cpu"):
    config_path = os.path.join(model_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    checkpoint_path = os.path.join(model_dir, model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_obj(config.model.class_name)(**config.model.params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def load_class_names(encoding_json_path: str):
    with open(encoding_json_path, "r") as f:
        label_encoding = json.load(f)
    class_names = [label_encoding[str(i)] for i in range(len(label_encoding))]
    return class_names

def load_saved_dataset(dataset_path: str):
    dataset = LandmarkDataset.load(dataset_path)
    return dataset

def create_dataset_without_augmentations(model_dir: str, split: str, seed: int = None):
    config_path = os.path.join(model_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    augmentations = {
        "train": None,
        "test": None,
    }
    dataset = LandmarkDataset(config.dataset, config.features, augmentations, split, seed=seed)
    return dataset

def create_cross_validation_datasets(train_dataset: LandmarkDataset, epoch: int, k_folds: int = 5):
    datasets = {}
    
    val_dataset = copy.deepcopy(train_dataset)
    val_dataset.dataset_split = "val"
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")
    if len(train_dataset) != len(val_dataset):
        raise ValueError("Training and validation datasets don't have the same length")
    
    # Create groups array where each sample from the same video gets the same group number
    groups = []
    labels = []
    for video_idx in range(len(train_dataset.metadata)):
        video_label = train_dataset.metadata.iloc[video_idx]["label_encoded"]
        groups.extend([video_idx] * train_dataset.samples_per_video[video_idx])
        labels.extend([video_label] * train_dataset.samples_per_video[video_idx])
    
    groups = np.array(groups)
    stratified_group_kfold = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42 + epoch)
    fold_indices = list(stratified_group_kfold.split(range(len(train_dataset)), labels, groups=groups))

    for fold, (train_ids, val_ids) in enumerate(fold_indices):
        datasets[fold] = (Subset(train_dataset, train_ids), Subset(val_dataset, val_ids))
    return datasets

def test_model_on_dataset(model: nn.Module, dataset: LandmarkDataset, class_names: List[str], \
               device: str = "cpu", model_name: str = "best_model.pt", return_eval: bool = False, \
               majority_ensemble: bool = False, logits_ensemble: bool = False, confidence_ensemble: bool = False):
    """
    Test model on dataset.
    
    Args:
        model: Model to use for inference
        dataset: Dataset to test on
        class_names: List of class names
        device: Device to use for inference (default: 'cpu')
        model_name: Name of model checkpoint file to load (default: 'best_model.pt')
        majority_ensemble: Whether to use majority ensemble (default: False)
        logits_ensemble: Whether to use logits ensemble (default: False)
        confidence_ensemble: Whether to use confidence ensemble (default: False)
    """
    start_time = time.time()
    results = {}

    print("Loading inference engines...")
    sample_inference = InferenceEngine(
        model=model,
        device=device,
        ensemble_strategy=None
    )

    print("Running sample inference...")
    sample_preds, sample_labels, sample_probs = sample_inference.predict(dataset, return_labels=True, return_full_probs=True)
    
    sample_eval = EvaluationMetrics(
        sample_preds,
        sample_labels,
        sample_probs,
        num_classes=len(class_names),
        class_names=class_names
    )

    results['sample_acc'] = sample_eval.accuracy
    results['sample_loss'] = sample_eval.get_loss(nn.CrossEntropyLoss())
    results['sample_topk_acc_2'] = sample_eval.get_topk_accuracy(2)
    results['sample_topk_acc_3'] = sample_eval.get_topk_accuracy(3)
    results['sample_topk_acc_4'] = sample_eval.get_topk_accuracy(4)
    results['sample_topk_acc_5'] = sample_eval.get_topk_accuracy(5)

    if majority_ensemble:
        print("Running majority ensemble inference...")
        majority_inference = InferenceEngine(
            model=model,
            device=device,
            ensemble_strategy='majority'
        )
        majority_output = majority_inference.predict(dataset, return_labels=True, return_full_probs=True)
        majority_preds, majority_labels, majority_probs = [np.array(x) for x in zip(*majority_output.values())]
        
        majority_eval = EvaluationMetrics(
            majority_preds,
            majority_labels,
            majority_probs,
            num_classes=len(class_names),
            class_names=class_names
        )
        results['majority_acc'] = majority_eval.accuracy
        results['majority_loss'] = majority_eval.get_loss(nn.CrossEntropyLoss())

    if logits_ensemble:
        print("Running logits ensemble inference...")
        logits_inference = InferenceEngine(
            model=model,
            device=device,
            ensemble_strategy='logits_average'
        )
        logits_output = logits_inference.predict(dataset, return_labels=True, return_full_probs=True)
        logits_preds, logits_labels, logits_probs = [np.array(x) for x in zip(*logits_output.values())]
        
        logits_eval = EvaluationMetrics(
            logits_preds,
            logits_labels,
            logits_probs,
            num_classes=len(class_names),
            class_names=class_names
        )
        results['logits_acc'] = logits_eval.accuracy
        results['logits_loss'] = logits_eval.get_loss(nn.CrossEntropyLoss())

    if confidence_ensemble:
        print("Running confidence ensemble inference...")
        confidence_inference = InferenceEngine(
            model=model,
            device=device,
            ensemble_strategy='confidence_weighted'
        )
        confidence_output = confidence_inference.predict(dataset, return_labels=True, return_full_probs=True)
        confidence_preds, confidence_labels, confidence_probs = [np.array(x) for x in zip(*confidence_output.values())]
        
        confidence_eval = EvaluationMetrics(
            confidence_preds,
            confidence_labels,
            confidence_probs,
            num_classes=len(class_names),
            class_names=class_names
        )
        results['confidence_acc'] = confidence_eval.accuracy
        results['confidence_loss'] = confidence_eval.get_loss(nn.CrossEntropyLoss())
    print(f"Done ({time.time() - start_time} seconds)")
    if return_eval:
        return results, sample_eval
    else:
        return results

def test_model_on_cross_validation_datasets(model: nn.Module, datasets: Dict[int, Tuple[Subset, Subset]], class_names: List[str], \
               device: str = "cpu", model_name: str = "best_model.pt", \
               majority_ensemble: bool = False, logits_ensemble: bool = False, confidence_ensemble: bool = False):
    """
    Test model on cross validation datasets.
    """
    train_results = {}
    val_results = {}
    avg_results = {}

    for fold, (train_dataset, val_dataset) in datasets.items():
        print(f"- Fold {fold + 1} -")
        train_results[fold] = test_model_on_dataset(model, train_dataset, class_names, device, model_name, \
                                              majority_ensemble, logits_ensemble, confidence_ensemble)
        val_results[fold] = test_model_on_dataset(model, val_dataset, class_names, device, model_name, \
                                              majority_ensemble, logits_ensemble, confidence_ensemble)
        
    avg_results['train_acc'] = np.mean([results['sample_acc'] for results in train_results.values()])
    avg_results['train_loss'] = np.mean([results['sample_loss'] for results in train_results.values()])
    avg_results['val_acc'] = np.mean([results['sample_acc'] for results in val_results.values()])
    avg_results['val_loss'] = np.mean([results['sample_loss'] for results in val_results.values()])
    
    return avg_results, train_results, val_results