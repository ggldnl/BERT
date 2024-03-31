from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import config
import utils
import os


class CornellMovieDialogsDataset(Dataset):

    def __init__(self,
                 data,
                 max_seq_len,
                 tokenizer=None,
                 ):

        super(CornellMovieDialogsDataset, self).__init__()

        self.data = data  # List of (question, answer) pairs
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        pass


class CornellMovieDialogsDataModule(pl.LightningDataModule):
    """
    Custom PyTorch Lightning DataModule class. The datamodule will
    download the content at the url only if the required file does
    not exist. This datamodule implements the logic to handle the
    Foursquare dataset.

    This dataset contains over 220,000 conversational exchanges
    between more than 10,000 pairs of characters in various movies and TV shows.

    More information here:
    https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
    """

    def __init__(self,
                 data_dir,
                 encoder_seq_len,
                 tokenizer=None,
                 download='infer',  # This string can either be 'yes', 'no', 'infer'
                 random_split=True,
                 batch_size=config.BATCH_SIZE,
                 num_workers=config.NUM_WORKERS
                 ):
        super(CornellMovieDialogsDataModule, self).__init__()

        self.data_dir = data_dir
        self.encoder_seq_len = encoder_seq_len
        self.tokenizer = tokenizer
        self.download = download.lower()
        self.random_split = random_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.resource_url = r'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
        self.conversations_path = os.path.join(self.data_dir, 'cornell movie-dialogs corpus/movie_conversations.txt')
        self.lines_path = os.path.join(self.data_dir, 'cornell movie-dialogs corpus/movie_lines.txt')

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.raw_data = None

    def prepare_data(self):

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        if self.download == 'infer':
            if any(not os.path.exists(path) for path in [self.conversations_path, self.lines_path]):
                self.download = 'yes'

        if self.download == 'yes':
            zip_path = os.path.join(self.data_dir, 'data.zip')
            print(f'Downloading resource...')
            utils.download_resource(self.resource_url, zip_path)
            print(f'Resource downloaded.\nExtracting resource...')
            utils.extract_zip(zip_path, self.data_dir)
            print(f'Resource extracted.\nRemoving zip file...')
            os.remove(zip_path)
            print(f'Done.')
        else:
            print(f'resource already downloaded.\nDone.')

    def train_text_val_split(self, data, train_percent, val_percent):

        # Compute the sizes of each set
        num_samples = len(data)
        num_train = int(train_percent * num_samples)
        num_val = int(val_percent * num_samples)

        # Shuffle the indices to randomly select samples for each set
        indices = np.arange(num_samples)
        if self.random_split:
            np.random.shuffle(indices)

        # Split the indices into training, validation, and test sets
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        # Create lists for each set using the selected indices
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]

        return train_data, test_data, val_data

    def setup(self, stage=None):

        # Open the  conversations file
        with open(self.conversations_path, 'r', encoding='iso-8859-1') as c:
            conversations = c.readlines()

        # Open the lines file
        with open(self.lines_path, 'r', encoding='iso-8859-1') as l:
            lines = l.readlines()

        # Splitting the lines using the marker
        lines_dic = {}
        for line in lines:
            objects = line.split(" +++$+++ ")
            lines_dic[objects[0]] = objects[-1]

        # Generate question answer pairs
        pairs = []
        for con in conversations:
            ids = eval(con.split(" +++$+++ ")[-1])
            for i in range(len(ids)):
                qa_pairs = []

                if i == len(ids) - 1:
                    break

                first = lines_dic[ids[i]].strip()
                second = lines_dic[ids[i + 1]].strip()

                qa_pairs.append(' '.join(first.split()[:self.encoder_seq_len]))
                qa_pairs.append(' '.join(second.split()[:self.encoder_seq_len]))
                pairs.append(qa_pairs)

        self.raw_data = pairs

        train_data, test_data, val_data = self.train_text_val_split(self.raw_data, 0.7, 0.1)
        self.train_dataset = CornellMovieDialogsDataset(
            train_data,
            self.encoder_seq_len,
            self.tokenizer,
        )

        self.test_dataset = CornellMovieDialogsDataset(
            test_data,
            self.encoder_seq_len,
            self.tokenizer
        )

        self.val_dataset = CornellMovieDialogsDataset(
            val_data,
            self.encoder_seq_len,
            self.tokenizer
        )

    def as_dataset(self):
        # Train + test + val dataset used to train the tokenizer
        return CornellMovieDialogsDataset(
            self.raw_data,
            self.encoder_seq_len,
            self.tokenizer
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


if __name__ == '__main__':

    from masked_language_modeling.tokenizer import Tokenizer

    # Load a tokenizer if present
    tokenizer_path = os.path.join(config.TOK_DIR, r'tokenizer.txt')

    # Check if a tokenizer backup exists
    if os.path.exists(tokenizer_path):
        print(f'Loading tokenizer...')
        tokenizer = Tokenizer.load(tokenizer_path, driver='txt')
    else:
        print(f'Creating tokenizer...')
        tokenizer = None

    datamodule = CornellMovieDialogsDataModule(
        config.DATA_DIR,
        config.SEQ_LEN,
        tokenizer=tokenizer,
        download='infer',
        random_split=False
    )
    datamodule.prepare_data()  # Download the data
    datamodule.setup()  # Setup it

    # Check that the size of the tensors are right
    dataloader = datamodule.as_dataset()

    """
    for batch in dataloader:

        if tokenizer is not None:
            encoder_input = batch['encoder_input']
            decoder_input = batch['decoder_input']
            encoder_mask = batch['encoder_mask']
            decoder_mask = batch['decoder_mask']
            label = batch['label']

            print(f'Encoder input (batch) size  : {encoder_input.shape}')
            print(f'Decoder input (batch) size  : {decoder_input.shape}')
            print(f'Encoder mask (batch) size   : {encoder_mask.shape}')
            print(f'Decoder mask (batch) size   : {decoder_mask.shape}')
            print(f'Label (batch) size          : {label.shape}')

        else:
            sequence = batch['sequence']
            pois = [elem[0] for elem in sequence]
            timestamps = [elem[1] for elem in sequence]

            print(f'Sequence    : {sequence}')
            print(f'POI list    : {pois}')
            print(f'Timestamps  : {timestamps}')

        break
    """
