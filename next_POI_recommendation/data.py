from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import random
import config
import utils
import torch
import os


class FoursquareDataset(Dataset):

    def __init__(self,
                 data,
                 max_seq_len,
                 tokenizer=None,
                 mask_percent=0.15,
                 discard_non_mask_indexes=True
                 ):
        """
        The transformer is a sequence-to-sequence model used for translation
        from a language to another. The two languages might use a different
        set of tokens.
        """

        super(FoursquareDataset, self).__init__()

        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mask_percent = mask_percent
        self.discard_non_mask_indexes = discard_non_mask_indexes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        seq_dict = self.data[item]

        if self.tokenizer is None:
            return seq_dict  # Contains poi_sequence, timestamp_sequence, is_next boolean

        # Convert the sequence into tokens and then into ids
        sequence_tokens = self.tokenizer.sequence_to_tokens(seq_dict['poi_sequence'])
        sequence_ids = self.tokenizer.tokens_to_ids(sequence_tokens)

        # Split the sequence_ids into the first sequence and the second sequence (NSP)
        first_part = sequence_ids[:-1]
        second_part = sequence_ids[-1:]

        # Create the bert input
        bert_input = (
                [self.tokenizer.cls_token_id] +
                first_part +
                [self.tokenizer.sep_token_id] +
                second_part +
                [self.tokenizer.sep_token_id]
        )

        # Create a copy to use as label
        bert_label = bert_input.copy()

        # Create the segment label
        segment_label = (
            [1] * (len(first_part) + 2) +
            [2] * (len(second_part) + 1)
        )

        # Switch to tensors
        bert_input = torch.tensor(bert_input, dtype=torch.int64)
        bert_label = torch.tensor(bert_label, dtype=torch.int64)
        segment_label = torch.tensor(segment_label, dtype=torch.int64)
        is_next = torch.tensor([seq_dict['is_next']], dtype=torch.bool)

        # Apply the mask
        rand = torch.rand(bert_input.shape)
        mask_arr = (
            (rand < self.mask_percent) *                        # Indexes for masked tokens
            (bert_input != self.tokenizer.cls_token_id) *  # Avoid placing a mask on the cls token
            (bert_input != self.tokenizer.sep_token_id)    # Avoid placing a mask on the sep token
        )
        selection = torch.flatten(mask_arr.nonzero()).tolist()

        # We can choose to discard or keep the indices of the non-masked tokens
        if self.discard_non_mask_indexes:
            bert_label = torch.zeros_like(bert_input)
            bert_label[selection] = bert_input[selection]

        bert_input[selection] = self.tokenizer.msk_token_id

        # Add the padding
        num_pad_tokens = self.max_seq_len - bert_input.shape[0]
        pad_tokens = [self.tokenizer.pad_token_id] * num_pad_tokens
        padding = torch.tensor(pad_tokens, dtype=torch.int64)

        bert_input = torch.cat([
            bert_input,
            padding
        ])
        bert_label = torch.cat([
            bert_label,
            padding
        ])
        segment_label = torch.cat([
            segment_label,
            padding
        ])

        return {
            'is_next': is_next,
            'bert_input': bert_input,
            'bert_label': bert_label,
            'segment_label': segment_label,
        }


class FoursquareDataModule(pl.LightningDataModule):
    """
    Custom PyTorch Lightning DataModule class. The datamodule will
    download the content at the url only if the required file does
    not exist. This datamodule implements the logic to handle the
    Foursquare dataset.

    This dataset contains check-ins in NYC and Tokyo collected
    for about 10 month (from 12 April 2012 to 16 February 2013).

    More information here:
    https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_ID_46
    """

    def __init__(self,
                 data_dir,
                 max_seq_len,
                 min_seq_len,
                 encoder_seq_len,
                 use='both',  # This string can either be 'nyc', 'tky', 'both'
                 tokenizer=None,
                 download='infer',  # This string can either be 'yes', 'no', 'infer'
                 random_split=True,
                 batch_size=config.BATCH_SIZE,
                 num_workers=config.NUM_WORKERS
                 ):
        super(FoursquareDataModule, self).__init__()

        if use not in ['both', 'nyc', 'tky']:
            raise ValueError(f'Value for the \'use\' parameter can only be \'nyc\', \'tky\' or \'both\' to use' +
                             'data from New York, Tokyo or both of them.')

        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.encoder_seq_len = encoder_seq_len
        self.use = use.lower()
        self.tokenizer = tokenizer
        self.download = download.lower()
        self.random_split = random_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.resource_url = r'http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip'
        self.nyc_path = os.path.join(self.data_dir, 'dataset_tsmc2014/dataset_TSMC2014_NYC.txt')
        self.tky_path = os.path.join(self.data_dir, 'dataset_tsmc2014/dataset_TSMC2014_TKY.txt')
        self.schema = [
            'User ID',
            'Venue ID',
            'Venue category ID',
            'Venue category name',
            'Latitude',
            'Longitude',
            'Timezone',
            'UTC time'
        ]

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.raw_data = None

    def prepare_data(self):

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        if self.download == 'infer':
            if any(not os.path.exists(path) for path in [self.nyc_path, self.tky_path]):
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

    def train_test_val_split(self, data, train_percent, val_percent):

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

    @staticmethod
    def filter_data(
            data_source,
            min_user_count_per_poi=10,
            min_poi_count_per_user=20,
            max_poi_count_per_user=50,
    ):

        # Filter POIs with unique visitors count less than min_user_count_per_poi
        venue_user_counts = data_source.groupby('Venue ID')['User ID'].nunique().reset_index(name='user_count')

        # Filter Venue IDs with user counts less than min_user_count_per_poi
        selected_venues = venue_user_counts[venue_user_counts['user_count'] >= min_user_count_per_poi]

        # Extract relevant rows from the original NYC dataset for the selected venues
        filtered_data = data_source[data_source['Venue ID'].isin(selected_venues['Venue ID'])]
        # Group by User ID and count unique Venue IDs
        user_visit_counts = filtered_data.groupby('User ID')['Venue ID'].nunique().reset_index(name='visit_count')

        # Filter users with visit counts between min_poi_count_per_user and max_poi_count_per_user
        selected_users = user_visit_counts[(user_visit_counts['visit_count'] >= min_poi_count_per_user) &
                                           (user_visit_counts['visit_count'] <= max_poi_count_per_user)]

        # Extract relevant rows from the original dataset for the selected users
        filtered_data = filtered_data[filtered_data['User ID'].isin(selected_users['User ID'])]

        # Sort it
        filtered_data['UTC time'] = pd.to_datetime(filtered_data['UTC time'], format='%a %b %d %H:%M:%S +0000 %Y')

        return filtered_data.sort_values(by=['User ID', 'UTC time'], ascending=[True, True])

    @staticmethod
    def create_user_poi_sequence_dict(data):

        # Initialize dictionaries to store sequences
        lu_dict = {}

        # Iterate through the sorted DataFrame to create sequences
        for user_id, group in data.groupby('User ID'):
            lu_sequence = [(venue_id, str(timestamp)) for venue_id, timestamp in
                           zip(group['Venue ID'], group['UTC time'])]

            lu_dict[user_id] = lu_sequence

        return lu_dict

    @staticmethod
    def create_poi_sequence_list(data_dict):
        return [seq for user, seq in data_dict.items()]

    def split_sequences_exceeding_len(self, data):
        result_list = []
        for sublist in data:
            if len(sublist) > self.max_seq_len:

                # Split the sublist into sublists with a maximum length
                splitted_sublist = [sublist[i:i + self.max_seq_len] for i in range(0, len(sublist), self.max_seq_len)]

                # Add all the resulting sublists except the last one, for which we need to check the length
                result_list.extend(splitted_sublist[:-1])

                if len(splitted_sublist[-1]) > self.min_seq_len:
                    result_list.append(splitted_sublist[-1])
            else:

                # If the sublist is within the maximum length, add it as is
                result_list.append(sublist)

        return result_list

    @staticmethod
    def shuffle_target_pois(data, permute_probability=0.5):

        # We will permute two elements for each index we generate,
        # so we halve the permute probability
        permute_probability = permute_probability / 2

        # Select the indexes of the elements to permute
        elements_to_permute = [i for i in range(len(data)) if random.random() < permute_probability]

        for index in elements_to_permute:
            # Select a different element to swap with
            other_index = random.choice([i for i in range(len(data)) if i != index])

            # Swap the last element of the #index sequence with the last element of the #other_index sequence
            data[index]['poi_sequence'][-1], data[other_index]['poi_sequence'][-1] = (
                data[other_index]['poi_sequence'][-1], data[index]['poi_sequence'][-1]
            )

            # Do the same for the timestamp
            data[index]['timestamp_sequence'][-1], data[other_index]['timestamp_sequence'][-1] = (
                data[other_index]['timestamp_sequence'][-1], data[index]['timestamp_sequence'][-1]
            )

            # Update the control boolean for the pair of sequences
            data[index]['is_next'] = False
            data[other_index]['is_next'] = False

    def setup(self, stage=None):

        # raw_data contains the entries in the input files:
        # 'User ID',
        # 'Venue ID',
        # 'Venue category ID',
        # 'Venue category name',
        # 'Latitude',
        # 'Longitude',
        # 'Timezone',
        # 'UTC time'
        # data will instead contain the (POI, timestamp) sequences for each user

        if self.use == 'nyc':
            raw_data = utils.read_tsv(self.nyc_path, skip_header=False, encoding='latin-1')
        elif self.use == 'tky':
            raw_data = utils.read_tsv(self.tky_path, skip_header=False, encoding='latin-1')
        else:  # self.use == 'both'
            nyc_data = utils.read_tsv(self.nyc_path, skip_header=False, encoding='latin-1')
            tky_data = utils.read_tsv(self.tky_path, skip_header=False, encoding='latin-1')
            raw_data = nyc_data + tky_data

        # Create a dataframe (this will simplify things later)
        df = pd.DataFrame(raw_data, columns=self.schema)

        # Filter the data
        filtered_df = self.filter_data(df)

        # Build the sequences
        data_dict = self.create_user_poi_sequence_dict(filtered_df)

        # Discard the user and create a list of POI sequences
        data = self.create_poi_sequence_list(data_dict)

        # Split sequences that exceed max length
        data = self.split_sequences_exceeding_len(data)

        # Transform the list of (poi, timestamp) sequences into a list of dictionaries;
        # each dictionary will have:
        # - poi sequence
        # - timestamp sequence
        # - boolean value that will tell us if the last poi is the true one or has been changed
        data = [{
            'poi_sequence': [elem[0] for elem in sequence],
            'timestamp_sequence': [elem[1] for elem in sequence],
            'is_next': True
        } for sequence in data]

        # Shuffle part of the data for the "next sentence prediction" task
        # (it won't exactly be a next sentence prediction task but something similar)
        self.shuffle_target_pois(data, 0.5)

        self.raw_data = data

        train_data, test_data, val_data = self.train_test_val_split(data, 0.7, 0.1)
        self.train_dataset = FoursquareDataset(
            train_data,
            self.encoder_seq_len,
            self.tokenizer,
        )

        self.test_dataset = FoursquareDataset(
            test_data,
            self.encoder_seq_len,
            self.tokenizer
        )

        self.val_dataset = FoursquareDataset(
            val_data,
            self.encoder_seq_len,
            self.tokenizer
        )

    def sequences_dataset(self):
        # Train + test + val dataset used to train the tokenizer
        return FoursquareDataset(
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

    from next_POI_recommendation.tokenizer import POITokenizer

    # Load a tokenizer if present
    tokenizer_path = os.path.join(config.TOK_DIR, r'poi_tokenizer.txt')

    # Check if a tokenizer backup exists
    if os.path.exists(tokenizer_path):
        print(f'Loading tokenizer...')
        tokenizer = POITokenizer.load(tokenizer_path, driver='txt')
    else:
        tokenizer = None

    datamodule = FoursquareDataModule(
        config.DATA_DIR,
        config.MAX_SEQ_LEN,
        config.MIN_SEQ_LEN,
        config.SEQ_LEN,
        tokenizer=tokenizer,
        download='infer',
        random_split=False
    )
    datamodule.prepare_data()  # Download the data
    datamodule.setup()  # Setup it

    # Check that the size of the tensors are right
    train_dataloader = datamodule.train_dataloader()

    for batch in train_dataloader:

        if tokenizer is not None:

            # poi_sequence = batch['poi_sequence']
            # timestamp_sequence = batch['timestamp_sequence']
            is_next = batch['is_next']
            bert_input = batch['bert_input']
            bert_label = batch['bert_label']
            segment_label = batch['segment_label']

            print(f'Bert input (batch)          : {bert_input.shape}')
            print(f'Bert label (batch)          : {bert_label.shape}')
            print(f'Segment label (batch)       : {segment_label.shape}')

        else:

            poi_sequence = batch['poi_sequence']
            timestamp_sequence = batch['timestamp_sequence']
            is_next = batch['is_next']

            print(f'POI sequence        : {poi_sequence}')
            print(f'Timestamp sequence  : {timestamp_sequence}')
            print(f'Is next             : {is_next}')

        break
