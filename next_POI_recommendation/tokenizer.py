import pickle
import os


class POITokenizer:
    """
    Custom tokenizer for the Next POI recommendation task.
    """

    def __init__(self,
                 pad_token='[PAD]',
                 cls_token='[CLS]',
                 sep_token='[SEP]',
                 msk_token='[MSK]',
                 unk_token='[UNK]',
                 min_frequency=1
                 ):

        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.msk_token = msk_token
        self.unk_token = unk_token

        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.msk_token_id = 3
        self.unk_token_id = 4

        self.special_tokens = [pad_token, cls_token, sep_token, msk_token, unk_token]
        self.special_tokens_ids = [0, 1, 2, 3, 4]

        self.min_frequency = min_frequency
        self.poi2index = {}
        self.index2poi = {}
        self.vocab_size = 0

    def train(self, pois):

        # Compute the frequency for each token
        token_frequency = {}
        for token in pois:
            if token in token_frequency:
                token_frequency[token] += 1
            else:
                token_frequency[token] = 1

        # Remove tokens with low frequency
        unique_tokens = [token for token, freq in token_frequency.items() if freq >= self.min_frequency]

        # Populate poi2index dictionary
        self.poi2index = {poi: idx + len(self.special_tokens) for idx, poi in enumerate(sorted(unique_tokens))}

        # Add the special tokens
        for token, token_id in zip(self.special_tokens, self.special_tokens_ids):
            self.poi2index[token] = token_id

        # Populate index2poi dictionary
        self.index2poi = {idx: poi for poi, idx in self.poi2index.items()}

        self.vocab_size = len(self.poi2index)

    def get_vocab_size(self):
        return self.vocab_size

    def token_to_id(self, token):
        return self.poi2index[token]

    def id_to_token(self, token_id):
        return self.index2poi[token_id]

    def sequence_to_tokens(self, sequence):
        return [token if token in self.poi2index else self.unk_token for token in sequence]

    def tokens_to_ids(self, tokens):
        return [self.poi2index[token] for token in tokens]

    def to_pickle(self, path):

        # Get the directory part of the file path
        parent_folder = os.path.dirname(path)

        # Create the parent folders if they don't exist
        os.makedirs(parent_folder, exist_ok=True)

        with open(path, 'wb') as file:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'poi2index': self.poi2index,
                'index2poi': self.index2poi
            }, file)

    def from_pickle(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.vocab_size = data['vocab_size']
            self.poi2index = data['poi2index']
            self.index2poi = data['index2poi']

    def to_txt(self, path):

        # Get the directory part of the file path
        parent_folder = os.path.dirname(path)

        # Create the parent folders if they don't exist
        os.makedirs(parent_folder, exist_ok=True)

        with open(path, 'w') as file:
            file.write(f"vocab_size: {self.vocab_size}\n")
            for word, index in self.poi2index.items():
                file.write(f"{word}\t{index}\n")

    def from_txt(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            vocab_size = int(lines[0].split(':')[1])
            self.vocab_size = vocab_size

            poi2index = {}
            for line in lines[1:]:
                word, index = line.strip().split('\t')
                poi2index[word] = int(index)
            self.poi2index = poi2index

            index2poi = {index: word for word, index in poi2index.items()}
            self.index2poi = index2poi

    @classmethod
    def load(cls, path, driver='pkl'):

        driver = driver.lower()

        if driver == 'infer':
            driver = path.split('.')[-1]

        if driver not in ['pkl', 'pickle', 'txt']:
            raise ValueError(f'Invalid driver: {driver}')

        tokenizer = POITokenizer()

        if driver == 'pkl' or driver == 'pickle':
            tokenizer.from_pickle(path)
        else:
            tokenizer.from_txt(path)

        return tokenizer


if __name__ == '__main__':

    from data import FoursquareDataModule
    import config

    def create_tokenizer():

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

        # Take all the sequences and extract the pois
        sequences = datamodule.sequences_dataset()
        pois = [elem for sequence in sequences for elem in sequence['poi_sequence']]

        # Build a tokenizer from the pois
        tokenizer = POITokenizer()
        tokenizer.train(pois)

        return tokenizer

    # We use the same tokenizer for both the input and the output sequence
    tokenizer_path = os.path.join(config.TOK_DIR, r'poi_tokenizer.txt')

    # Check if a tokenizer backup exists
    if os.path.exists(tokenizer_path):
        print(f'Loading tokenizer...')
        tokenizer = POITokenizer.load(tokenizer_path, driver='txt')
    # If not, create it
    else:
        print(f'Creating tokenizer...')
        tokenizer = create_tokenizer()
        tokenizer.to_txt(tokenizer_path)
        print(f'Tokenizer saved to {tokenizer_path}')

    print(f'Tokenizer vocabulary size: {tokenizer.vocab_size}')
