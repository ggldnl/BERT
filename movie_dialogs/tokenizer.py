from nltk.tokenize import word_tokenize
import random
import pickle
import torch
import nltk
import os


class BERTTokenizer:
    """
    Custom implementation of the WordPiece tokenizer. The code has been
    adapted from the one provided in the huggingface library.
    More information here:
    https://huggingface.co/learn/nlp-course/en/chapter6/6
    """

    def __init__(self,
                 vocab_size=70,
                 pad_token='[PAD]',
                 cls_token='[CLS]',
                 sep_token='[SEP]',
                 msk_token='[MSK]',
                 unk_token='[UNK]',
                 min_frequency=2,
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
        self.vocab_size = vocab_size

        self.alphabet = None
        self.vocabulary = None
        self.token2index = None
        self.index2token = None

        # Download NLTK stopwords and punkt tokenizer if we haven't already
        nltk.download('stopwords')
        nltk.download('punkt')

    @staticmethod
    def tokenize_sentence(sentence):
        tokens = [token.lower() for token in word_tokenize(sentence)]
        return tokens

    @staticmethod
    def compute_pair_scores(splits, word_frequencies):
        letter_freqs = {}
        pair_freqs = {}

        for word, freq in word_frequencies.items():
            split = splits[word]
            if len(split) == 1:

                if split[0] in letter_freqs:
                    letter_freqs[split[0]] += freq
                else:
                    letter_freqs[split[0]] = freq

                continue

            for i in range(len(split) - 1):

                pair = (split[i], split[i + 1])

                if split[i] in letter_freqs:
                    letter_freqs[split[i]] += freq
                else:
                    letter_freqs[split[i]] = freq

                if pair in pair_freqs:
                    pair_freqs[pair] += freq
                else:
                    pair_freqs[pair] = freq

            if split[-1] in letter_freqs:
                letter_freqs[split[-1]] += freq
            else:
                letter_freqs[split[-1]] = freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    @staticmethod
    def merge_pair(a, b, splits, word_frequencies):
        for word in word_frequencies:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits

    def train(self, corpus):

        # First, we need to pre-tokenize the corpus into words.
        all_tokens = [token.lower() for sentence in corpus for token in self.tokenize_sentence(sentence)]

        # Compute the frequency for each token
        token_frequencies = {}
        for token in all_tokens:
            if token in token_frequencies:
                token_frequencies[token] += 1
            else:
                token_frequencies[token] = 1

        # Remove tokens with low frequency
        # unique_tokens = [token for token, freq in token_frequencies.items() if freq >= self.min_frequency]

        # the alphabet is the unique set composed of all the first letters of words,
        # and all the other letters that appear in words prefixed by ##
        alphabet = []
        for word in token_frequencies.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")

        self.alphabet = alphabet

        # WordPiece finds the longest subword that is in the vocabulary, then splits on it.
        self.vocabulary = self.special_tokens + alphabet.copy()

        # Next we split each word, with all the letters that are not the first prefixed by ##:
        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in token_frequencies.keys()
        }

        # Loop until we have learned all the merges we want
        while len(self.vocabulary) < self.vocab_size:
            scores = self.compute_pair_scores(splits, token_frequencies)
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            splits = self.merge_pair(*best_pair, splits, token_frequencies)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            self.vocabulary.append(new_token)

        # Populate word2index dictionary
        self.token2index = {word: idx for idx, word in enumerate(sorted(self.vocabulary))}

    def encode_word(self, word):
        # To tokenize a new text, we pre-tokenize it, split it,
        # then apply the tokenization algorithm on each word.
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocabulary:
                i -= 1
            if i == 0:
                return [self.unk_token]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(self, sentence):
        pre_tokenize_sentence = [token for token in self.tokenize_sentence(sentence)]
        encoded_words = [self.encode_word(word) for word in pre_tokenize_sentence]
        return sum(encoded_words, [])

    def token_to_id(self, token):
        return self.token2index[token] if token in self.token2index else self.unk_token_id

    def get_vocab_size(self):
        return self.vocab_size

    def to_pickle(self, path):

        # Get the directory part of the file path
        parent_folder = os.path.dirname(path)

        # Create the parent folders if they don't exist
        os.makedirs(parent_folder, exist_ok=True)

        with open(path, 'wb') as file:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'token2index': self.token2index,
                'index2token': self.index2token
            }, file)

    def from_pickle(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.vocab_size = data['vocab_size']
            self.token2index = data['token2index']
            self.index2token = data['index2token']

    def to_txt(self, path):

        # Get the directory part of the file path
        parent_folder = os.path.dirname(path)

        # Create the parent folders if they don't exist
        os.makedirs(parent_folder, exist_ok=True)

        with open(path, 'w') as file:
            file.write(f"vocab_size: {self.vocab_size}\n")
            for token, index in self.token2index.items():
                file.write(f"{token}\t{index}\n")

    def from_txt(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            vocab_size = int(lines[0].split(':')[1])
            self.vocab_size = vocab_size

            token2index = {}
            for line in lines[1:]:
                token, index = line.strip().split('\t')
                token2index[token] = int(index)
            self.token2index = token2index

            index2token = {index: token for token, index in token2index.items()}
            self.index2token = index2token

    @classmethod
    def load(cls, path, driver='pkl'):

        driver = driver.lower()

        if driver == 'infer':
            driver = path.split('.')[-1]

        if driver not in ['pkl', 'pickle', 'txt']:
            raise ValueError(f'Invalid driver: {driver}')

        tokenizer = BERTTokenizer()

        if driver == 'pkl' or driver == 'pickle':
            tokenizer.from_pickle(path)
        else:
            tokenizer.from_txt(path)

        return tokenizer


if __name__ == '__main__':

    import config
    tokenizer_path = os.path.join(config.TOK_DIR, r'bert_tokenizer_example.txt')

    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    tokenizer = BERTTokenizer()
    tokenizer.train(corpus)
    tokenizer.to_txt(tokenizer_path)

    sentence = "What am I even doing with my life? You will suffer"
    tokenized_sentence = tokenizer.tokenize(sentence)
    print(f'Original sentence : {sentence}')
    print(f'Tokenized sentence: {tokenized_sentence}')
