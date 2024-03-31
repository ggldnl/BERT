from architecture.blocks.multihead_attention import MultiHeadAttentionBlock
from architecture.blocks.positional_embeddings import PositionalEmbeddings
from architecture.blocks.input_embeddings import InputEmbeddings
from architecture.blocks.feed_forward import FeedForwardBlock
from architecture.blocks.projection import MLMProjectionLayer
from architecture.blocks.encoder_block import EncoderBlock
from architecture.blocks.encoder import Encoder
from architecture.cosine_warmup import CosineWarmupScheduler
from tokenizer import POITokenizer
from data import FoursquareDataModule
from architecture.model import BERT
from architecture.model import LightningBERT
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import config
import os


def create_tokenizer():

    # Build a datamodule WITHOUT tokenizers
    datamodule = FoursquareDataModule(
        config.DATA_DIR,
        config.MAX_SEQ_LEN,
        config.MIN_SEQ_LEN,
        config.SEQ_LEN,
        tokenizer=None,
        download='infer',
        random_split=False
    )
    datamodule.prepare_data()  # Download the data
    datamodule.setup()  # Setup it

    # Take the corpus (we need an iterable)
    train_dataloader = datamodule.sequences_dataset()
    pois = []
    for batch in train_dataloader:
        batch_poi_sequences = batch['poi_sequence']
        batch_pois = [poi for sequence in batch_poi_sequences for poi in sequence]
        pois.extend(batch_pois)

    # Use the corpus to train the tokenizer
    tokenizer = POITokenizer()
    tokenizer.train(pois)

    return tokenizer


if __name__ == '__main__':

    # Use txt for better interpretability
    tokenizer_path = os.path.join(config.TOK_DIR, r'poi_tokenizer.txt')

    # Check if a tokenizer backup exists
    if os.path.exists(tokenizer_path):
        print(f'Loading tokenizer...')
        tokenizer = POITokenizer.load(tokenizer_path, driver='txt')
    else:
        print(f'Creating tokenizer...')
        tokenizer = create_tokenizer()
        tokenizer.to_txt(tokenizer_path)

    print(f'Vocabulary size: {tokenizer.vocab_size}')
    print(f'-' * 100)

    print(f'Creating datamodule...')

    # Redefine the datamodule giving it the source and target tokenizers
    datamodule = FoursquareDataModule(
        config.DATA_DIR,
        config.MAX_SEQ_LEN,
        config.MIN_SEQ_LEN,
        config.SEQ_LEN,
        tokenizer=tokenizer,
        download='infer',
        random_split=False
    )

    print(f'Datamodule created.')
    print(f'-' * 100)

    print(f'Defining model...')

    # Initialize the model
    bert = BERT.build(
        tokenizer.vocab_size,
        config.SEQ_LEN,
        config.EMBED_DIM,
        config.NUM_ENCODERS,
        config.HEADS,
        config.DROPOUT,
        config.D_FF
    )

    # Criterion, Optimizer and Scheduler
    criterion = nn.NLLLoss(
        ignore_index=tokenizer.pad_token_id,
        # label_smoothing=0.1
    )
    optimizer = Adam(bert.parameters(), lr=1e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=2000)

    # Initialize the Lightning model
    model = LightningBERT(bert, criterion, optimizer, scheduler)

    print(f'Model defined.')
    print(f'-' * 100)

    # Initialize the Trainer
    trainer = pl.Trainer(max_epochs=config.NUM_EPOCHS)

    # Train and test the model
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
