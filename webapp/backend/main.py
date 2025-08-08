import logging
from fastapi import FastAPI, UploadFile, File, Depends
import numpy as np
from config import Config
from dependencies import get_inference_engine, get_feature_processor, get_sampling_func
from schemas import PredictionResponse
from utils import cleanup_files
from services import (
    PredictionService,
    ResultsSaver,
    FeaturePipeline,
    VideoValidator,
    VideoSaver,
    MetadataExtractor,
    VideoAnalyzerService,
    PreprocessorService,
    LandmarkDrawerService,
    MetadataFilterService
)


from services.model_references import FeatureProcessor
from services.model_references import InferenceEngine


logger = logging.getLogger(__name__)
app = FastAPI(title="Video Sign Language Prediction API")

@app.post("/predict/", response_model=PredictionResponse)
async def predict_video(
    file: UploadFile = File(...),
    config: Config = Depends(Config),
    inference_engine: InferenceEngine = Depends(get_inference_engine),
    feature_processor: FeatureProcessor = Depends(get_feature_processor),
    sampling_func: callable = Depends(get_sampling_func)
):
    # 1. Validate video
    VideoValidator.validate_extension(file.filename)

    # 2. Save file
    video_path = await VideoSaver.save(file, config.INTERIM_DIR)

    # 3. Extract metadata
    metadata_row = MetadataExtractor.extract(str(video_path))

    # 4. Analyze video
    analyzer = VideoAnalyzerService.analyze(metadata_row, config)

    # 5. Preprocess
    landmarks_path = PreprocessorService.preprocess(metadata_row, analyzer, config)

    # 6. Annotate video
    annotated_path = LandmarkDrawerService.draw(landmarks_path, metadata_row, config, file.filename)

    # 7. Sample & process features
    frames = np.load(landmarks_path, allow_pickle=True)
    sample_indices = sampling_func(
        num_frames=len(frames),
        params=config.config_yaml.dataset.frame_sampling_test.params
    )
    metadata_row_proc = MetadataFilterService.load_and_filter(config, file.filename)
    input_sample = FeaturePipeline.process(frames, sample_indices, metadata_row_proc, feature_processor)

    # 8. Predict
    prediction, probs, label_encoding = PredictionService.predict_and_format(
        input_sample, inference_engine, config.LABEL_ENCODING_PATH
    )

    output_video_path = str(config.OUTPUT_DIR / file.filename)
    # Upload skeleton video to Cloudinary
    cloud_url = ResultsSaver.upload_video_to_cloudinary(output_video_path)

    # 9. Save results
    ResultsSaver.save_txt(prediction, probs, label_encoding, file.filename, config.OUTPUT_DIR)

    # 10. Cleanup
    cleanup_files([video_path, landmarks_path,output_video_path])

    # 11. Response
    return PredictionResponse(
        prediction={"class_id": int(prediction), "label": label_encoding[str(int(prediction))]},
        probabilities={label_encoding[str(i)]: float(p) for i, p in enumerate(probs)},
        output_files={
            "skeleton_video": cloud_url,
            "results_txt": str(config.OUTPUT_DIR / file.filename.replace(".mp4", ".txt"))
        }
    )
