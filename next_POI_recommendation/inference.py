from architecture.model import BERT
from tokenizer import POITokenizer
from data import FoursquareDataModule
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
    print('Model created')

    # Optional: restore weights

    # Use the model to make inference
    # TODO