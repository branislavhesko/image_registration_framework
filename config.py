from enum import auto, Enum
from model.resunet import ResUNetBN2D


class Mode(Enum):
    TRAIN = auto()
    VALIDATE = auto()
    TEST = auto()


class Paths:
    DATASET = "RETINA"
    CHECKPOINT_PATH = "./ckpt"
    PATH_TO_IMAGES_FIRST = "./data/retina_dataset/fundus"
    PATH_TO_IMAGES_SECOND = "./data/retina_dataset/experimental"

class ModelConfiguration:
    IN_CHANNELS = 3
    INITIAL_LR = 1e-3
    LR_DECAY = 0.97
    MODEL = ResUNetBN2D
    MOMENTUM = 0.95
    NORMALIZE_FEATURES = True
    OUT_FEATURES = 64
    WEIGHT_DECAY = 1e-4


class Configuration(ModelConfiguration, Paths):
    MODES = [Mode.TRAIN, Mode.VALIDATE]

    BATCH_SIZE = {Mode.TRAIN: 1, Mode.VALIDATE: 1}
    EPOCHS = 1
    NEGATIVE_LOSS_COEF = 1.
    NUM_WORKERS = 4
    SHUFFLE = {Mode.TRAIN: False, Mode.VALIDATE: False}
    USE_CUDA = True


class PredictorConfiguration(Configuration):
    WEIGHTS_PAH = ""
