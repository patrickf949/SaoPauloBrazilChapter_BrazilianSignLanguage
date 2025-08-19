from enum import Enum
import logging
from typing import Literal, Optional
import numpy as np
from fastapi import APIRouter, Form, HTTPException, UploadFile, File, Depends, status
from config import Config
from dependencies import get_inference_engine, get_feature_processor, get_sampling_func
from schemas import PredictionResponse
from utils import cleanup_files
from services import (
    VideoValidator,
    RequestVideoSaver,
    LabelVideoSaver,
    MetadataExtractor,
    VideoAnalyzerService,
    PreprocessorService,
    LandmarkDrawerService,
    MetadataFilterService,
    FeaturePipeline,
    PredictionService,
    ResultsSaver
)
from services.model_references import InferenceEngine, FeatureProcessor
import datetime

router = APIRouter()


class AllowedLabels(str, Enum):
    ajudar_ne_1 = "ajudar_ne_1"
    animal_sb_2 = "animal_sb_2"
    aniversário_uf_3 = "aniversário_uf_3"
    ano_vl_6 = "ano_vl_6"
    banana_sb_2 = "banana_sb_2"
    banheiro_ne_1 = "banheiro_ne_1"
    bebê_uf_3 = "bebê_uf_3"
    cabeça_sb_2 = "cabeça_sb_2"
    café_vl_5 = "café_vl_5"
    carne_vl_5 = "carne_vl_5"
    casa_vl_4 = "casa_vl_4"
    cebola_ne_1 = "cebola_ne_1"
    comer_uf_3 = "comer_uf_3"
    cortar_vl_6 = "cortar_vl_6"
    crescer_ne_1 = "crescer_ne_1"
    família_uf_3 = "família_uf_3"
    filho_vl_6 = "filho_vl_6"
    garganta_sb_2 = "garganta_sb_2"
    homem_vl_5 = "homem_vl_5"
    jovem_ne_1 = "jovem_ne_1"
    ouvir_uf_3 = "ouvir_uf_3"
    pai_vl_4 = "pai_vl_4"
    sopa_sb_2 = "sopa_sb_2"
    sorvete_ne_1 = "sorvete_ne_1"
    vagina_uf_3 = "vagina_uf_3"


@router.post("/predict/", response_model=PredictionResponse)
async def predict_video(
    file: Optional[UploadFile] = File(None),
    label: Optional[AllowedLabels] = Form(None),
    config: Config = Depends(Config),
    inference_engine: InferenceEngine = Depends(get_inference_engine),
    feature_processor: FeatureProcessor = Depends(get_feature_processor),
    sampling_func: callable = Depends(get_sampling_func)
):
    try:
        # 2. Save video
        video_path = None
        file_name = None
        # Validation: ensure exactly one is provided
        if (file is None and label is None) or (file is not None and label is not None):

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You must provide either a file OR a label, but not both."
            )

        # Case 1: Video file provided
        if file is not None:
            # validate and process as before
            await VideoValidator.validate_file(file)
            video_path = await RequestVideoSaver.save(file, config.INTERIM_DIR)

            # ... rest of your pipeline
            # return PredictionResponse(...)

        # Case 2: Label provided
        if label is not None:
            # Here you might:
            video_path = await LabelVideoSaver.save(config.SAMPLE_VIDEOS / f"{label.value}.mp4", config.INTERIM_DIR)

        logging.warning(f"saved file in {video_path}")
        file_name = video_path.name
        # 3. Extract metadata
        metadata_row = MetadataExtractor.extract(str(video_path))

        # 4. Analyze
        analyzer = VideoAnalyzerService.analyze(metadata_row, config)

        # 5. Preprocess
        landmarks_path = PreprocessorService.preprocess(
            metadata_row, analyzer, config)

        # 6. Annotate
        annotated_path = LandmarkDrawerService.draw(
            landmarks_path, metadata_row, config, file_name)

        # 7. Sample & process features
        frames = np.load(landmarks_path, allow_pickle=True)
        sample_indices = sampling_func(
            num_frames=len(frames),
            params=config.config_yaml.dataset.frame_sampling_test.params
        )
        metadata_row_proc = MetadataFilterService.load_and_filter(
            config, file_name)
        input_sample = FeaturePipeline.process(
            frames, sample_indices, metadata_row_proc, feature_processor)

        # 8. Predict
        prediction, probs, label_encoding = PredictionService.predict_and_format(
            input_sample, inference_engine, config.LABEL_ENCODING_PATH
        )

        # 9. Save results
        output_video_path = str(config.OUTPUT_DIR / file_name)
        cloud_url = ResultsSaver.upload_video_to_cloudinary(output_video_path)
        ResultsSaver.save_txt(
            prediction, probs, label_encoding, file_name, config.OUTPUT_DIR)

        # 10. Cleanup
        cleanup_files([config.DATA_DIR])

        # 11. Response
        return PredictionResponse(
            prediction={"class_id": int(
                prediction), "label": label_encoding[str(int(prediction))]},
            probabilities={label_encoding[str(i)]: float(
                p) for i, p in enumerate(probs)},
            output_files={"skeleton_video": cloud_url}
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )


@router.get("/")
async def root():
    return f"Brazilian Sign Language Recognition API ✅.\nUTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
