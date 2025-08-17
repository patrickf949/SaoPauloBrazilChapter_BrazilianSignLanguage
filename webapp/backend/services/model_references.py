
# Add /code modules dir to sys.path so imports work
from pathlib import Path
import sys


current_dir = Path(__file__).resolve().parent.parent
root_dir = current_dir.parent.parent

# ------------------------------------------------------------
# if /webapp doesn't need to be standalone, uncomment this line to import directly from /code:
code_dir = root_dir / 'code'
# ------------------------------------------------------------
# if /webapp should be standalone, run sync_code.py script, and uncomment this line:
# code_dir = current_dir / 'shared_code'
print("=== sys.path BEFORE ===")
for p in sys.path:
    print(p)

print(f"code dir: {code_dir}")
if str(code_dir) not in sys.path:
    print(f"enters to insert code dir: {code_dir}")
    sys.path.insert(0, str(code_dir))
    print(f"sys.path: {sys.path[0]}")

print("\n=== sys.path AFTER ===")
for p in sys.path:
    print(p)
# Import /code modules. (Type ignore comments for Pylance)
from data.download_videos import get_video_metadata # type: ignore
from preprocess.video_analyzer import VideoAnalyzer # type: ignore
from preprocess.preprocessor import Preprocessor # type: ignore
from preprocess.vizualisation import draw_landmarks_on_video_with_frame # type: ignore
from model.features.feature_processor import FeatureProcessor # type: ignore
from model.utils.inference import InferenceEngine # type: ignore
from model.utils.utils import load_obj # type: ignore
from model.dataset.frame_sampling import get_sampling_function # type: ignore
# Import /code modules. (Type ignore comments for Pylance)
