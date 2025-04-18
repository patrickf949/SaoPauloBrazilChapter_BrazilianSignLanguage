from models.landmark.dataset.base_estimator import BaseEstimator


class ExperimentFeatureEstimator(BaseEstimator):
    """
    Compute any additional feature like:
        - hand in front of face (0/1 binary)
        - hands touch (0/1 binary)
        ...
    """

    def __init__(self): ...
