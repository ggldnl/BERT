# Model parameters
EMBED_DIM = 768
SEQ_LEN = 350
NUM_ENCODERS = 2  # Number of encoder blocks
DROPOUT = 0.1
HEADS = 8
D_FF = 1024

# Dataset parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
MIN_SEQ_LEN = 20  # Minimum number of pois to form a sequence
MAX_SEQ_LEN = 50  # Maximum number of pois to form a sequence

# Training parameters
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
PRECISION = '16-mixed'

# Folders
DATA_DIR = 'dataset'
TOK_DIR = 'tokenizers'
