import random
import pickle
import torch
import os


class POITokenizer:

    def __init__(self,
                 unk_token='[UNK]',
                 pad_token='[PAD]',
                 sos_token='[SOS]',
                 eos_token='[EOS]',
                 msk_token='[MSK]',
                 min_frequency=1
                 ):

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.msk_token = msk_token

        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.msk_token_id = 4

        self.special_tokens = [pad_token, sos_token, eos_token, unk_token, msk_token]
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

    def get_encoder_input(self, sequence, max_seq_len=None, mask_percent=0.0, mask_last_token=False):
        """
        Given a sequence, produces the input for the encoder. SOS and EOS tokens
        are added respectively to the start and the end of the sequence for the
        encoder input. The unknown token is used in place of unrecognized tokens
        and a padding token is added to match the max sequence length.
        """

        # Convert the sequence into tokens and then into ids
        sequence_tokens = self.sequence_to_tokens(sequence)
        sequence_ids = self.tokens_to_ids(sequence_tokens)

        if mask_percent > 0.0:
            mask_ids = random.sample(sequence_ids, int(len(sequence_ids) * mask_percent))
            for index in mask_ids:
                sequence_ids[index] = self.msk_token_id

        if mask_last_token:
            sequence_ids[-1] = self.msk_token_id

        if max_seq_len is None:
            max_seq_len = len(sequence_ids) + 2

        # Number of padding tokens (max length - number of tokens in the
        # sequence - SOS token - EOS token). If max_seq_len is not specified,
        # it is set to the length of the list of tokens + SOS and EOS and
        # enc_padding_tokens will be 0
        enc_padding_tokens = max_seq_len - len(sequence_ids) - 2

        encoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.int64),
            torch.tensor(sequence_ids, dtype=torch.int64),
            torch.tensor([self.eos_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * enc_padding_tokens, dtype=torch.int64)
        ])

        return encoder_input

    def get_decoder_input(self, sequence, max_seq_len=None):
        """
        Given a sequence, produces the input for the decoder. Only the SOS is
        added to the sequence. The unknown token is used in place of unrecognized
        tokens and a padding token is added to match the max sequence length.
        """

        # Convert the sequence into tokens and then into input ids
        sequence_tokens = self.sequence_to_tokens(sequence.lower())
        sequence_ids = self.tokens_to_ids(sequence_tokens)

        if max_seq_len is None:
            max_seq_len = len(sequence_ids) + 1

        # Number of padding tokens (max length - number of tokens in the
        # sequence - SOS token). If max_seq_len is not specified, it is set
        # to the length of the list of tokens + SOS and enc_padding_tokens
        # will be 0
        dec_padding_tokens = max_seq_len - len(sequence_ids) - 1

        decoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.int64),
            torch.tensor(sequence_ids, dtype=torch.int64),
            torch.tensor([self.pad_token_id] * dec_padding_tokens, dtype=torch.int64)
        ])

        return decoder_input

    def get_label(self, sequence, max_seq_len=None):
        """
        Given a sequence, produces tokenized version of the label. Only the EOS is
        added to the sequence. The unknown token is used in place of unrecognized
        tokens and a padding token is added to match the max sequence length.
        """

        # Convert the sequence into tokens and then into input ids
        sequence_tokens = self.sequence_to_tokens(sequence.lower())
        sequence_ids = self.tokens_to_ids(sequence_tokens)

        if max_seq_len is None:
            max_seq_len = len(sequence_ids) + 1

        # Number of padding tokens (max length - number of tokens in the
        # sequence - EOS token). If max_seq_len is not specified, it is set
        # to the length of the list of tokens + EOS and enc_padding_tokens
        # will be 0
        lab_padding_tokens = max_seq_len - len(sequence_ids) - 1

        label = torch.cat([
            torch.tensor(sequence_ids, dtype=torch.int64),
            torch.tensor([self.eos_token_id], dtype=torch.int64),
            torch.tensor([self.pad_token_id] * lab_padding_tokens, dtype=torch.int64)
        ])

        return label

    def get_encoder_mask(self, encoder_input):
        return (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).type(torch.int64)

    @staticmethod
    def causal_mask(size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int64)
        return mask == 0

    def get_decoder_mask(self, decoder_input):
        return (decoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).type(torch.int64) & self.causal_mask(decoder_input.size(0))

    def encode(self, sequence):
        """
        Given a sequence, returns a list of token ids. The unknown token
        is used in place of unrecognized tokens.
        """

        # Convert the sequence into tokens and then into input ids
        sequence_tokens = self.sequence_to_tokens(sequence)
        sequence_ids = self.tokens_to_ids(sequence_tokens)

        return sequence_ids

    def decode(self, token_ids):
        """
        Given a list of token ids, returns the respective sequence.
        """
        return [self.id_to_token(token_id) for token_id in token_ids]

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
            source_tokenizer=None,
            target_tokenizer=None,
            download='infer',
            random_split=False
        )
        datamodule.prepare_data()  # Download the data
        datamodule.setup()  # Setup it

        # Take all the sequences and extract the pois
        sequences = datamodule.sequences_dataset()
        pois = [elem for sequence in sequences for elem in sequence['pois']]

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
