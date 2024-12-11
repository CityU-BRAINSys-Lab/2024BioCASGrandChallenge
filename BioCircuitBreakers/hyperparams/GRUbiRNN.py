DATASET_NUM_STEPS = 7
DATASET_BIN_WIDTH = 50 * 4e-3
DATASET_LABEL_SERIES = False

DATALOADER_SHUFFLE = [True, False]
DATALOADER_BATCH_SIZE = [256, 256]

MODEL_GRU_LAYER_DIMS = [32]
MODEL_OUTPUT_SERIES = DATASET_LABEL_SERIES
MODEL_OUTPUT_DIM = 2

PREPROCESSORS = ["Normalize"]
AUGMENTATORS = ["Jittering", "GaussianNoise"]

TRAIN_NUM_EPOCHS = 20
# TRAIN_KLDIV_LOSS_WEIGHT = 0.4
TRAIN_LOG_TRANSFORM = True
TRAIN_FOCUS_FACTOR = 1.5

TEST_DATALOADER_BATCH_SIZE = 256
TEST_NETWORK = ""

TESTMODELADD_BIN_WIDTH = DATASET_BIN_WIDTH
TESTMODELADD_NUM_STEPS = DATASET_NUM_STEPS
TESTMODELADD_LOG_TRAIN = TRAIN_LOG_TRANSFORM
TESTMODELADD_FOCUS_FACTOR = TRAIN_FOCUS_FACTOR

OTHER_INFO = "always zero initial hidden state, corrected reverse order, 2layer FCs"