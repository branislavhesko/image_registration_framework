from enum import auto, Enum

from model.resunet import ResUNetBN2F


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
    INITIAL_LR = 1e-4
    LR_DECAY = 0.97
    MODEL = ResUNetBN2F
    MOMENTUM = 0.95
    NORMALIZE_FEATURES = True
    OUT_FEATURES = 3
    WEIGHT_DECAY = 1e-4


class LossConfiguration:
    NUM_POS_PAIRS = 128
    NUM_NEG_PAIRS = 64
    PIXEL_LIMIT = 10
    NUM_NEG_INDICES_FOR_LOSS = 5
    NEGATIVE_LOSS_WEIGHT = 1


class Configuration(ModelConfiguration, Paths, LossConfiguration):
    MODES = [Mode.TRAIN, Mode.VALIDATE]

    BATCH_SIZE = {Mode.TRAIN: 1, Mode.VALIDATE: 1}
    DISTANCE_LIMIT = 10
    EPOCHS = 30
    NEGATIVE_LOSS_COEF = -1.
    NUM_WORKERS = 0
    SHUFFLE = {Mode.TRAIN: False, Mode.VALIDATE: False}
    USE_CUDA = True
    VALIDATION_FREQUENCY = 5
    TRAIN_VISUALIZATION_FREQUENCY = 10


class PredictorConfiguration(Configuration):
    WEIGHTS_PAH = ""
