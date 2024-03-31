from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import random
import config
import utils
import torch
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

        s1, s2, is_next = item

        if self.tokenizer is None:
            return {
                'first_sentence': s1,
                'second_sentence': s2
            }

        # Tokenize the sentences
        tokenized_s1 = self.tokenizer.tokenize(s1)
        tokenized_s2 = self.tokenizer.tokenize(s2)

        # Transform the sentences to lists of ids
        tokenized_s1_ids = [self.tokenizer.token_to_id(token) for token in tokenized_s1]
        tokenized_s2_ids = [self.tokenizer.token_to_id(token) for token in tokenized_s2]

        # Create the labels
        tokenized_s1_ids_label = [0 for _ in tokenized_s1_ids]
        tokenized_s2_ids_label = [0 for _ in tokenized_s2_ids]

        mask_percent = 0.15

        # Apply the mask to the first sentence
        s1_mask_ids = random.sample(tokenized_s1_ids, int(len(tokenized_s1_ids) * mask_percent))
        for index in s1_mask_ids:
            tokenized_s1_ids_label[index] = tokenized_s1_ids[index]
            tokenized_s1_ids[index] = self.tokenizer.msk_token_id

        # Apply the mask to the second sentence
        s2_mask_ids = random.sample(tokenized_s2_ids, int(len(tokenized_s2_ids) * mask_percent))
        for index in s2_mask_ids:
            tokenized_s2_ids_label[index] = tokenized_s2_ids[index]
            tokenized_s2_ids[index] = self.tokenizer.msk_token_id

        # Add CLS and EOS tokens
        bert_input = (
            [self.tokenizer.cls_token_id] +
            tokenized_s1_ids +
            [self.tokenizer.sep_token_id] +
            tokenized_s2_ids +
            [self.tokenizer.sep_token_id]
        )
        bert_label = (
            [self.tokenizer.pad_token_id] +
            tokenized_s1_ids_label +
            [self.tokenizer.pad_token_id] +
            tokenized_s2_ids_label +
            [self.tokenizer.pad_token_id]
        )
        segment_label = (
            [1 for _ in range(len(tokenized_s1_ids) + 2)] +
            [2 for _ in range(len(tokenized_s2_ids) + 1)]
        )

        # Number of padding tokens
        num_padding_tokens = self.max_seq_len - len(bert_input)

        bert_input = bert_input + [self.tokenizer.pad_token_id] * num_padding_tokens
        bert_label = bert_label + [self.tokenizer.pad_token_id] * num_padding_tokens

        return {
            'first_sentence': s1,
            'second_sentence': s2,
            'bert_input': torch.tensor(bert_input, dtype=torch.int64),
            'bert_label': torch.tensor(bert_label, dtype=torch.int64),
            'segment_label': torch.tensor(segment_label, dtype=torch.int64),
            'is_next_sentence': torch.tensor(is_next)
        }


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
            print(f'Resource already downloaded.')

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

                if i == len(ids) - 1:
                    break

                first = lines_dic[ids[i]].strip()
                second = lines_dic[ids[i + 1]].strip()

                a = ' '.join(first.split()[:self.encoder_seq_len])
                b = ' '.join(second.split()[:self.encoder_seq_len])
                pairs.append([a, b, True])

        # Now we can change some of the pairs in order to have positive and negative examples for the
        # next sentence prediction task

        # Perform the permutation
        permute_probability = 0.25
        elements_to_permute = [i for i in range(len(pairs)) if random.random() < permute_probability]

        # Perform permutation
        for index in elements_to_permute:

            # Select a different element to swap with
            other_index = random.choice([i for i in range(len(pairs)) if i != index])

            # Swap the elements
            pairs[index], pairs[other_index] = pairs[other_index], pairs[index]

            # Signal these two pairs are not original
            pairs[index][2] = False
            pairs[other_index][2] = False

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

    from movie_dialogs.tokenizer import BERTTokenizer

    # Load a tokenizer if present
    tokenizer_path = os.path.join(config.TOK_DIR, r'tokenizer.txt')

    # Check if a tokenizer backup exists
    if os.path.exists(tokenizer_path):
        print(f'Loading tokenizer...')
        tokenizer = BERTTokenizer.load(tokenizer_path, driver='txt')
    else:
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

    # The datamodule is an iterable containing a set of (s1, s2, is_next) tuples.
    # We flatten the (s1, s2) into a single list for each element of the datamodule
    # discarding the last boolean value of each tuple
    sentences = [item for sublist in dataloader for item in sublist[:2]]

    # Build a tokenizer
    tokenizer = BERTTokenizer(1000)
    print(f'Training tokenizer...')
    tokenizer.train(sentences)
    tokenizer.to_txt(os.path.join(config.TOK_DIR, 'bert_tokenizer.txt'))

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
