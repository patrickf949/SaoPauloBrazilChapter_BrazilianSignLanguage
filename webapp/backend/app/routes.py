@app.post("/predict/", response_model=PredictionResponse)
async def predict_video(
    file: UploadFile = File(...),
    config: Config = Depends(Config),
    inference_engine: InferenceEngine = Depends(get_inference_engine),
    feature_processor: FeatureProcessor = Depends(get_feature_processor),
    sampling_func: callable = Depends(get_sampling_func)
):
    """
    Endpoint to upload a video file, process it, and return sign language predictions.
    """
    try:
        # Validate file extension
        if not file.filename.lower().endswith('.mp4'):
            raise HTTPException(status_code=400, detail="Only .mp4 files are supported")

        # Save uploaded file
        video_path = await save_uploaded_file(file, config.INTERIM_DIR)
        logger.info(f"Uploaded video saved to {video_path}")

        # Get video metadata
        metadata = get_video_metadata(str(video_path))
        if not metadata:
            raise HTTPException(status_code=500, detail="Failed to extract video metadata")
        metadata['data_source'] = 'app'
        metadata_row = pd.Series(metadata)
        logger.info(f"Video metadata: {metadata}")

        # Analyze video
        analyzer = VideoAnalyzer(
            metadata_row,
            config.TIMESTAMP,
            str(config.BASE_DIR),
            verbose=False,
            motion_detection_version=config.MOTION_VERSION,
            pose_detection_version=config.POSE_VERSION
        )
        analyzer.pose_detect()
        analyzer.pose_analyze()
        analyzer.motion_detect()
        motion_analysis = analyzer.motion_analyze()
        analysis_info = analyzer.save_analysis_info()
        logger.info("Video analysis completed")

        # Update preprocessing parameters
        preprocess_params = config.PREPROCESSING_PARAMS.copy()
        preprocess_params.update({
            "start_frame": analysis_info['motion_analysis']['start_frame'],
            "end_frame": analysis_info['motion_analysis']['end_frame'],
        })

        # Preprocess video
        preprocessor = Preprocessor(
            metadata_row,
            preprocess_params,
            str(config.BASE_DIR),
            config.MOTION_VERSION,
            config.POSE_VERSION,
            config.PREPROCESS_VERSION,
            verbose=False,
            save_intermediate=False,
        )
        landmarks_path = preprocessor.preprocess_landmarks()
        logger.info(f"Preprocessed landmarks saved to {landmarks_path}")

        # Generate skeleton video
        landmarks = np.load(landmarks_path, allow_pickle=True)
        output_video_path = config.OUTPUT_DIR / file.filename
        annotated_frames = draw_landmarks_on_video_with_frame(
            results_list=landmarks,
            output_path=str(output_video_path),
            fps=metadata_row['fps']
        )
        logger.info(f"Skeleton video saved to {output_video_path}")

        # Process features
        frames = np.load(landmarks_path, allow_pickle=True)
        sample_indices = sampling_func(
            num_frames=len(frames),
            params=config.config_yaml.dataset.frame_sampling_test.params
        )

        # Load preprocessed metadata
        metadata_csv_path = config.PREPROCESSED_DIR / f"landmarks_metadata_{config.PREPROCESS_VERSION}.csv"
        if not metadata_csv_path.exists():
            raise HTTPException(status_code=500, detail="Metadata CSV not found")
        preprocessed_metadata = pd.read_csv(metadata_csv_path)
        metadata_row_filtered = preprocessed_metadata.loc[
            preprocessed_metadata['filename'] == file.filename
        ]

        if len(metadata_row_filtered) == 0:
            raise HTTPException(status_code=404, detail=f"No metadata found for video {file.filename}")
        if len(metadata_row_filtered) > 1:
            raise HTTPException(status_code=500, detail=f"Multiple metadata entries found for video {file.filename}")

        metadata_row_proc = metadata_row_filtered.iloc[0].copy()
        metadata_row_proc['filename'] = metadata_row_proc['filename'].replace(".mp4", ".npy")

        sample_feature_tensors = []
        for sample_idx_list in sample_indices:
            sample_features = feature_processor.process_frames(
                frames, sample_idx_list, metadata_row_proc, 'test'
            )
            sample_feature_tensors.append(sample_features)
        input_sample = sample_feature_tensors[0]
        logger.info("Feature processing completed")

        # Perform inference
        prediction, probs = inference_engine.predict(
            inputs=input_sample,
            return_full_probs=True
        )
        logger.info(f"Prediction: {prediction}, Probabilities: {probs.tolist()}")

        # Format probabilities
        with open(config.LABEL_ENCODING_PATH, "r") as f:
            label_encoding = json.load(f)
        probs_dict = {label_encoding[str(i)]: float(prob) for i, prob in enumerate(probs)}
        probs_str = ", ".join(f"{label}: {prob:.3f}" for label, prob in probs_dict.items())

        # Save results to text file
        output_txt_path = config.OUTPUT_DIR / file.filename.replace(".mp4", ".txt")
        with open(output_txt_path, "w") as f:
            f.write(f"prediction: \n\tclass {prediction}: {label_encoding[str(int(prediction))]}\n")
            f.write(f"probabilities: {probs_str}\n")
        logger.info(f"Results saved to {output_txt_path}")

        # Clean up interim files
        cleanup_files([video_path, landmarks_path])
        logger.info("Interim files cleaned up")

        # Prepare response
        return {
            "prediction": {
                "class": int(prediction),
                "label": label_encoding[str(int(prediction))]
            },
            "probabilities": probs_dict,
            "output_files": {
                "skeleton_video": str(output_video_path),
                "results_txt": str(output_txt_path)
            }
        }

    except HTTPException as e:
        logger.error(f"HTTP error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")