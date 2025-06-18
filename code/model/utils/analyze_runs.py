import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

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

def test_model(model_dir: str, device: str = "cpu", seed: int = None, model_name: str = "best_model.pt"):
    """Test model on data.
    
    Args:
        model_dir: Directory containing model files and config
        device: Device to use for testing (default: 'cuda')
        seed: Random seed for reproducibility (optional)
        model_name: Name of model checkpoint file to load (default: 'best_model.pt')
    """
    config_path = os.path.join(model_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    checkpoint_path = os.path.join(model_dir, model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_obj(config.model.class_name)(**config.model.params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    augmentations = {
    "train": None,
    "test": None,
    }

    base_paths = load_base_paths()
    metadata_path = base_paths.metadata_base
    with open(os.path.join(metadata_path, "label_encoding.json"), "r") as f:
        label_encoding = json.load(f)
    class_names = [label_encoding[str(i)] for i in range(len(label_encoding))]

    print("Loading datasets...")
    train_dataset = LandmarkDataset(
        config.dataset, config.features, augmentations, "train", seed=seed
    )
    test_dataset = LandmarkDataset(
        config.dataset, config.features, augmentations, "test", seed=seed
    )

    print("Loading inference engines...")
    sample_inference = InferenceEngine(
        model=model,
        device='cpu',
        ensemble_strategy=None
    )
    majority_inference = InferenceEngine(
        model=model,
        device='cpu',
        ensemble_strategy='majority'
    )
    logits_inference = InferenceEngine(
        model=model,
        device='cpu',
        ensemble_strategy='logits_average'
    )
    confidence_inference = InferenceEngine(
        model=model,
        device='cpu',
        ensemble_strategy='confidence_weighted'
    )

    print("Predicting on train dataset...")
    print("\tBy sample")
    sample_train_preds, sample_train_labels, sample_train_probs = sample_inference.predict(train_dataset, return_labels=True, return_full_probs=True)
    print("\tBy majority")
    majority_train_output = majority_inference.predict(train_dataset, return_labels=True, return_full_probs=True)
    majority_train_preds, majority_train_labels, majority_train_probs = [np.array(x) for x in zip(*majority_train_output.values())]
    print("\tBy logits")
    logits_train_output = logits_inference.predict(train_dataset, return_labels=True, return_full_probs=True)
    logits_train_preds, logits_train_labels, logits_train_probs = [np.array(x) for x in zip(*logits_train_output.values())]
    print("\tBy confidence")
    confidence_train_output = confidence_inference.predict(train_dataset, return_labels=True, return_full_probs=True)
    confidence_train_preds, confidence_train_labels, confidence_train_probs = [np.array(x) for x in zip(*confidence_train_output.values())]

    print("Predicting on test dataset...")
    print("\tBy sample")
    sample_test_preds, sample_test_labels, sample_test_probs = sample_inference.predict(test_dataset, return_labels=True, return_full_probs=True)
    print("\tBy majority")
    majority_test_output = majority_inference.predict(test_dataset, return_labels=True, return_full_probs=True)
    majority_test_preds, majority_test_labels, majority_test_probs = [np.array(x) for x in zip(*majority_test_output.values())]
    print("\tBy logits")
    logits_test_output = logits_inference.predict(test_dataset, return_labels=True, return_full_probs=True)
    logits_test_preds, logits_test_labels, logits_test_probs = [np.array(x) for x in zip(*logits_test_output.values())]
    print("\tBy confidence")
    confidence_test_output = confidence_inference.predict(test_dataset, return_labels=True, return_full_probs=True)
    confidence_test_preds, confidence_test_labels, confidence_test_probs = [np.array(x) for x in zip(*confidence_test_output.values())]

    print("Evaluating predictions...")
    sample_train_eval = EvaluationMetrics(
        sample_train_preds,
        sample_train_labels,
        sample_train_probs,
        num_classes=len(class_names),
        class_names=class_names
    )

    majority_train_eval = EvaluationMetrics(
        majority_train_preds,
        majority_train_labels,
        majority_train_probs,
        num_classes=len(class_names),
        class_names=class_names
    )

    logits_train_eval = EvaluationMetrics(
        logits_train_preds,
        logits_train_labels,
        logits_train_probs,
        num_classes=len(class_names),
        class_names=class_names
    )

    confidence_train_eval = EvaluationMetrics(
        confidence_train_preds,
        confidence_train_labels,
        confidence_train_probs,
        num_classes=len(class_names),
        class_names=class_names
    )

    sample_test_eval = EvaluationMetrics(
        sample_test_preds,
        sample_test_labels,
        sample_test_probs,
        num_classes=len(class_names),
        class_names=class_names
    )

    majority_test_eval = EvaluationMetrics(
        majority_test_preds,
        majority_test_labels,
        majority_test_probs,
        num_classes=len(class_names),
        class_names=class_names
    )

    logits_test_eval = EvaluationMetrics(
        logits_test_preds,
        logits_test_labels,
        logits_test_probs,
        num_classes=len(class_names),
        class_names=class_names
    )

    confidence_test_eval = EvaluationMetrics(
        confidence_test_preds,
        confidence_test_labels,
        confidence_test_probs,
        num_classes=len(class_names),
        class_names=class_names
    )

    train_results = {
        'sample_acc': sample_train_eval.accuracy,
        'sample_loss': sample_train_eval.get_loss(nn.CrossEntropyLoss()),
        'majority_acc': majority_train_eval.accuracy,
        'majority_loss': majority_train_eval.get_loss(nn.CrossEntropyLoss()),
        'logits_acc': logits_train_eval.accuracy,
        'logits_loss': logits_train_eval.get_loss(nn.CrossEntropyLoss()),
        'confidence_acc': confidence_train_eval.accuracy,
        'confidence_loss': confidence_train_eval.get_loss(nn.CrossEntropyLoss()),
        'sample_topk_acc_2': sample_train_eval.get_topk_accuracy(2),
        'sample_topk_acc_3': sample_train_eval.get_topk_accuracy(3),
        'sample_topk_acc_4': sample_train_eval.get_topk_accuracy(4),
        'sample_topk_acc_5': sample_train_eval.get_topk_accuracy(5),
    }

    test_results = {
        'sample_acc': sample_test_eval.accuracy,
        'sample_loss': sample_test_eval.get_loss(nn.CrossEntropyLoss()),
        'majority_acc': majority_test_eval.accuracy,
        'majority_loss': majority_test_eval.get_loss(nn.CrossEntropyLoss()),
        'logits_acc': logits_test_eval.accuracy,
        'logits_loss': logits_test_eval.get_loss(nn.CrossEntropyLoss()),
        'confidence_acc': confidence_test_eval.accuracy,
        'confidence_loss': confidence_test_eval.get_loss(nn.CrossEntropyLoss()),
        'sample_topk_acc_2': sample_test_eval.get_topk_accuracy(2),
        'sample_topk_acc_3': sample_test_eval.get_topk_accuracy(3),
        'sample_topk_acc_4': sample_test_eval.get_topk_accuracy(4),
        'sample_topk_acc_5': sample_test_eval.get_topk_accuracy(5),
    }

    return train_results, test_results