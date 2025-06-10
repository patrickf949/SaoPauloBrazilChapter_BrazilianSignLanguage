import os
import sys
import json
import numpy as np
import yaml
import torch
import pandas as pd
from omegaconf import OmegaConf

def process_video(video_fn):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    code_dir = os.path.join(root_dir, 'code')
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    from data.download_videos import get_video_metadata
    from preprocess.video_analyzer import VideoAnalyzer
    from preprocess.preprocessor import Preprocessor
    from preprocess.vizualisation import draw_landmarks_on_video_with_frame
    from model.utils.utils import load_config, load_obj
    from model.dataset import frame_sampling
    from model.features.feature_processor import FeatureProcessor
    from model.utils.inference import InferenceEngine

    # Preprocessing
    ## Settings
    timestamp = "00"
    motion_version = "v0"
    pose_version = "v0"
    preprocess_version = "v0"

    path_to_root = os.path.join(root_dir, "webapp", "backend")

    video_path = os.path.join(path_to_root, "data", "interim", "RawCleanVideos", video_fn)
    print(video_path)
    metadata = get_video_metadata(video_path)
    print(metadata)
    metadata['data_source'] = 'app'
    metadata_row = pd.Series(metadata)

    ## Analyze video
    analyzer = VideoAnalyzer(
        metadata_row,
        timestamp,
        path_to_root,
        verbose=False,
        motion_detection_version=motion_version,
        pose_detection_version=pose_version
    )
    pose_data = analyzer.pose_detect()
    pose_analysis = analyzer.pose_analyze()
    motion_data = analyzer.motion_detect()
    motion_analysis = analyzer.motion_analyze()
    analysis_info = analyzer.save_analysis_info()


    ## Preprocess video
    preprocessing_params = {
        "face_width_aim": 0.155,
        "shoulders_width_aim": 0.35,
        "face_midpoint_to_shoulders_height_aim": 0.275,
        "shoulders_y_aim": 0.52,
        "use_statistic": "mean",
        "use_stationary_frames": True,
        "skip_stationary_frames": False,
        "start_frame": analysis_info['motion_analysis']['start_frame'],
        "end_frame": analysis_info['motion_analysis']['end_frame'],
    }
    preprocessor = Preprocessor(
        metadata_row,
        preprocessing_params,
        path_to_root,
        motion_version,
        pose_version,
        preprocess_version,
        verbose=False,
        save_intermediate=False,
    )
    landmarks_path = preprocessor.preprocess_landmarks()

    # Outputs
    ## Make Skeleton Video

    landmarks = np.load(landmarks_path, allow_pickle=True)
    output_path = path_to_root + f'/data/output/{metadata_row["filename"]}'
    annotated_frames = draw_landmarks_on_video_with_frame(results_list = landmarks, output_path = output_path, fps = metadata_row['fps'])

    ## Model Prediction
    ### Load config
    config_path = os.path.join(path_to_root, "model", "config.yaml")
    config = OmegaConf.load(config_path)
    sampling_func = frame_sampling.get_sampling_function(config.dataset.frame_sampling_test.method)
    sampling_params = config.dataset.frame_sampling_test.params

    ### Load all frames and sample them
    frames = np.load(landmarks_path, allow_pickle=True)
    sample_indices = sampling_func(
        num_frames=len(frames),
        params=sampling_params
    )
    ### Process Features
    preprocessed_dir = os.path.join(path_to_root, "data", "preprocessed")
    preprocessed_metadata = pd.read_csv(os.path.join(preprocessed_dir, f"landmarks_metadata_{preprocess_version}.csv"))

    preprocessed_metadata_row = preprocessed_metadata.loc[preprocessed_metadata['filename'] == video_fn]
    if len(preprocessed_metadata_row) == 0:
        raise ValueError(f"No metadata row found for video {video_fn}")
    if len(preprocessed_metadata_row) > 1:
        raise ValueError(f"Multiple metadata rows found for video {video_fn}")
    metadata_row = preprocessed_metadata_row.iloc[0].copy()
    metadata_row['filename'] = metadata_row['filename'].replace(".mp4", ".npy")


    feature_processor = FeatureProcessor(
        dataset_split='test',
        dataset_config=config.dataset,
        features_config=config.features,
        augmentation_config=config.augmentation,
        landmarks_dir=os.path.join(preprocessed_dir, "landmarks", preprocess_version),
    )

    sample_feature_tensors = []
    for sample_idx_list in sample_indices:
        sample_feature_tensors.append(feature_processor.process_frames(frames, sample_idx_list, metadata_row, 'test'))

    input_sample = sample_feature_tensors[0]

    ### Load model
    checkpoint = torch.load(os.path.join(path_to_root, "model", "model_checkpoint.pt"), map_location='cpu')
    model = load_obj(config.model.class_name)(**config.model.params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cpu')

    ### Inference
    inference_engine = InferenceEngine(
        model=model,
        device='cpu',
        ensemble_strategy=None
    )

    # input a single tensor and use _predict_single internally
    prediction, probs = inference_engine.predict(inputs = input_sample, return_full_probs=True )
    label_encoding_path = os.path.join(path_to_root, "model", "label_encoding.json")
    with open(label_encoding_path, "r") as f:
        label_encoding = json.load(f)
    probs_str = ""
    for i, prob in enumerate(probs):
        probs_str += f"\t{label_encoding[str(i)]}: {prob:.3f}, "

    # save to txt file
    with open(os.path.join(path_to_root, "data", "output", video_fn.replace(".mp4", ".txt")), "w") as f:
        f.write(f"prediction: \n\tclass {prediction}: {label_encoding[str(int(prediction))]}\n")
        f.write(f"probabilities: {probs_str}\n")

    return prediction, probs_str, label_encoding

if __name__ == "__main__":
    print("--------------------")
    # take video_fn input from command line
    video_fn = sys.argv[1]
    prediction, probs_str, label_encoding = process_video(video_fn)
    print(f"prediction: \n\tclass {prediction}: {label_encoding[str(int(prediction))]}")
    print(f"probabilities:")
    print(probs_str)
