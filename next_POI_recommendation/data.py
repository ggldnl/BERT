from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import config
import utils
import os


class FoursquareDataset(Dataset):

    def __init__(self,
                 data,
                 max_seq_len,
                 tokenizer=None,
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        poi_timestamp_sequence = self.data[item]
        pois = [elem[0] for elem in poi_timestamp_sequence]
        timestamps = [elem[1] for elem in poi_timestamp_sequence]

        if self.tokenizer is None:
            return {
                "sequence": poi_timestamp_sequence
            }

        # Tokenize the sequences and convert them to ids
        encoder_input = self.tokenizer.get_encoder_input(poi_sequence, self.max_seq_len, mask_percent=0.15)
        decoder_input = self.tokenizer.get_decoder_input(poi_sequence, self.max_seq_len)
        encoder_mask = self.tokenizer.get_encoder_mask(encoder_input)
        decoder_mask = self.tokenizer.get_decoder_mask(decoder_input)
        label = self.tokenizer.get_label(poi_sequence, self.max_seq_len)

        if self.source_time_slotter is None and self.target_time_slotter is None:
            return {
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'encoder_mask': encoder_mask,
                'decoder_mask': decoder_mask,
                'label': label,
                "sequence": poi_timestamp_sequence,
            }

        """
        encoder_time_slots = self.source_time_slotter.get_slots(timestamp_sequence, self.max_seq_len)
        decoder_time_slots = self.target_time_slotter.get_slots(timestamp_sequence, self.max_seq_len)

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_time_slots': encoder_time_slots,
            'decoder_time_slots': decoder_time_slots,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': label,
            "sequence": poi_timestamp_sequence,
            "pois": poi_sequence,
            "timestamps": timestamp_sequence,
        }
        """


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
    tokenizer_path = os.path.join(config.TOK_DIR, r'tokenizer.txt')

    # Check if a tokenizer backup exists
    if os.path.exists(tokenizer_path):
        print(f'Loading tokenizer...')
        tokenizer = POITokenizer.load(tokenizer_path, driver='txt')
    else:
        tokenizer = None

    """
    # Load a discretizer if present
    discretizer_path = os.path.join(config.DISC_DIR, r'discretizer.txt')

    # Check if a discretizer backup exists
    if os.path.exists(discretizer_path):
        print(f'Loading discretizer...')
        discretizer = Discretizer.load(discretizer_path, driver='txt')
    else:
        discretizer = None
    """

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
    train_dataloader = datamodule.sequences_dataset()

    for batch in train_dataloader:

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
