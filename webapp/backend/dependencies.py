from __future__ import annotations
from fastapi import Depends
from config import Config
import torch
import logging
from typing import Callable
import os
import sys
from pathlib import Path

from services.model_references import (
    load_obj,
    get_sampling_function,
    InferenceEngine,
    FeatureProcessor,
    current_dir
)


logger = logging.getLogger(__name__)


def get_inference_engine(config: Config = Depends(Config.get_instance)) -> InferenceEngine:
    """
    Provide an InferenceEngine instance based on configuration.

    Args:
        config (Config): Application configuration.

    Returns:
        InferenceEngine: Configured inference engine.
    """
    try:
        model_class = load_obj(config.config_yaml.model.class_name)
        model = model_class(**config.config_yaml.model.params)
        checkpoint_path = config.CHECKPOINT_PATH
        model.load_state_dict(torch.load(
            checkpoint_path, map_location='cpu')['model_state_dict'])
        model.eval()
        inference_engine = InferenceEngine(
            model=model,
            device='cpu',
            ensemble_strategy=config.config_yaml.get('ensemble_strategy', None)
        )
        logger.info("Initialized InferenceEngine")
        return inference_engine
    except Exception as e:
        logger.error(f"Failed to initialize InferenceEngine: {str(e)}")
        raise


def get_feature_processor(config: Config = Depends(Config.get_instance)) -> FeatureProcessor:
    """
    Provide a FeatureProcessor instance based on configuration.

    Args:
        config (Config): Application configuration.

    Returns:
        FeatureProcessor: Configured feature processor.
    """
    try:
        feature_processor = FeatureProcessor(
            dataset_split='test',
            dataset_config=config.config_yaml.dataset,
            features_config=config.config_yaml.features,
            augmentation_config=config.config_yaml.augmentation,
            landmarks_dir=os.path.join(
                current_dir, 'data', 'preprocessed', 'landmarks', 'v0'),
            seed=0
        )
        logger.info("Initialized FeatureProcessor")
        return feature_processor
    except Exception as e:
        logger.error(f"Failed to initialize FeatureProcessor: {str(e)}")
        raise


def get_sampling_func(config: Config = Depends(Config.get_instance)) -> Callable:
    """
    Provide a frame sampling function based on configuration.

    Args:
        config (Config): Application configuration.

    Returns:
        Callable: Frame sampling function.
    """
    try:
        sampling_func = get_sampling_function(
            config.config_yaml.dataset.frame_sampling_test.method)
        logger.info(
            f"Initialized sampling function: {config.config_yaml.dataset.frame_sampling_test.method}")
        return sampling_func
    except Exception as e:
        logger.error(f"Failed to initialize sampling function: {str(e)}")
        raise
