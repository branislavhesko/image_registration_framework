from config import PredictorConfiguration


class FeaturePredictor:

    def __init__(self, cfg: PredictorConfiguration):
        self._cfg = cfg
        self._cuda = cfg.USE_CUDA

    def load_model(self):
        pass