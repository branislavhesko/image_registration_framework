from enum import auto, Enum

from model.resunet import ResUNetBN2C


class Mode(Enum):
    TRAIN = auto()
    VALIDATE = auto()
    TEST = auto()


class Paths:
    DATASET = "RETINA"
    CHECKPOINT_PATH = "./ckpt"
    PATH_TO_IMAGES_FIRST = "./data/retina_dataset/experimental"
    PATH_TO_IMAGES_SECOND = "./data/retina_dataset/experimental"
    PATH_TO_MASKS = "./data/retina_dataset/experimental_segmentation"
    EXTENSION_FIRST = "tif"
    EXTENSION_MASK = "png"


class ModelConfiguration:
    IN_CHANNELS = 3
    INITIAL_LR = 1e-3
    LR_DECAY = 0.97
    MODEL = ResUNetBN2C
    MOMENTUM = 0.95
    NORMALIZE_FEATURES = False
    OUT_FEATURES = 32
    WEIGHT_DECAY = 1e-4


class LossConfiguration:
    NUM_POS_PAIRS = 4096
    NUM_NEG_PAIRS = 4096
    PIXEL_LIMIT = 10
    NUM_NEG_INDICES_FOR_LOSS = 5
    NEGATIVE_LOSS_WEIGHT = -2


class Configuration(ModelConfiguration, Paths, LossConfiguration):
    MODES = [Mode.TRAIN, Mode.VALIDATE]

    BATCH_SIZE = {Mode.TRAIN: 1, Mode.VALIDATE: 1}
    DISTANCE_LIMIT = 10
    EPOCHS = 30
    NEGATIVE_LOSS_COEF = 1.
    NUM_WORKERS = 0
    SHUFFLE = {Mode.TRAIN: True, Mode.VALIDATE: True}
    USE_CUDA = True
    VALIDATION_FREQUENCY = 5
    TRAIN_VISUALIZATION_FREQUENCY = 10


class PredictorConfiguration(Configuration):
    WEIGHTS_PAH = ""


class VesselConfiguration(Configuration):
    DATASET = "VESSEL"


class ArteryVeinConfiguration(Configuration):
    DATASET = "ARTERY_VEIN"
    EPOCHS = 200
    NORMALIZE_FEATURES = True
    PATH_TO_IMAGES_FIRST = "./data/vessel_artery_dataset/"
    NUM_NEG_PAIRS = 4096
    NUM_POS_PAIRS = 4096
    NEG_LOSS_CONSTANT = 1.
