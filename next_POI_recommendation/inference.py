from architecture.blocks.multihead_attention import MultiHeadAttentionBlock
from architecture.blocks.positional_encoding import PositionalEncoding
from architecture.blocks.input_embeddings import InputEmbeddings
from architecture.blocks.feed_forward import FeedForwardBlock
from architecture.blocks.projection import ProjectionLayer
from architecture.blocks.encoder_block import EncoderBlock
from architecture.blocks.decoder_block import DecoderBlock
from architecture.blocks.encoder import Encoder
from architecture.blocks.decoder import Decoder
from architecture.model import Transformer
from tokenizer import WordLevelTokenizer
from data import OPUSDataModule
import torch.nn as nn
import config
import os


def create_tokenizer(stage='source'):  # stage = ['source', 'target']

    # Build a datamodule WITHOUT tokenizers
    datamodule = OPUSDataModule(
        config.DATA_DIR,
        max_seq_len=config.MAX_SEQ_LEN,
        download='infer',
        random_split=False
    )
    datamodule.prepare_data()  # Download the data
    datamodule.setup()  # Setup it

    # Take the corpus (we need an iterable)
    train_dataloader = datamodule.train_dataloader()
    corpus = []
    for batch in train_dataloader:
        corpus.extend(batch[f'{stage}_text'])

    # Use the corpus to train the tokenizer
    tokenizer = WordLevelTokenizer()
    tokenizer.train(corpus)

    return tokenizer


def create_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        embed_dim: int = config.EMBED_DIM,
        num_encoders: int = config.NUM_ENCODERS,
        num_decoders: int = config.NUM_DECODERS,
        heads: int = config.HEADS,
        dropout: float = config.DROPOUT,
        d_ff: int = config.D_FF
):

    # creating embedding layers
    src_embed = InputEmbeddings(embed_dim, src_vocab_size)  # source vocab size to embedding size vectors
    tgt_embed = InputEmbeddings(embed_dim, tgt_vocab_size)  # target vocab size to embedding size vectors

    # creating positional encoding layers
    src_pos = PositionalEncoding(embed_dim, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(embed_dim, tgt_seq_len, dropout)

    # creating EncoderBlocks
    encoder_blocks = []
    for _ in range(num_encoders):
        encoder_self_attention_block = MultiHeadAttentionBlock(embed_dim, heads, dropout)  # self-attention
        feed_forward_block = FeedForwardBlock(embed_dim, d_ff, dropout)  # feedforward

        # combine layers into an EncoderBlock
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)  # appending EncoderBlock to the list of EncoderBlocks

    # creating decoder blocks
    decoder_blocks = []
    for _ in range(num_decoders):
        decoder_self_attention_block = MultiHeadAttentionBlock(embed_dim, heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(embed_dim, heads, dropout)  # cross-attention
        feed_forward_block = FeedForwardBlock(embed_dim, d_ff, dropout)  # feedforward

        # combining layers into a DecoderBlock
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)  # appending DecoderBlock and DecoderBlocks lists

    # creating the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Creating projection layer
    projection_layer = ProjectionLayer(embed_dim, tgt_vocab_size)

    # crating the transformer by combining everything above
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Assembled and initialized Transformer, Ready to be trained and validated!
    return transformer


if __name__ == '__main__':

    # Use txt for better interpretability
    source_tokenizer_path = os.path.join(config.TOK_DIR, r'tokenizer_source.txt')
    target_tokenizer_path = os.path.join(config.TOK_DIR, r'tokenizer_target.txt')

    # Check if a tokenizer backup exists
    if os.path.exists(source_tokenizer_path):
        print(f'Loading source tokenizer...')
        source_tokenizer = WordLevelTokenizer.load(source_tokenizer_path, driver='txt')
    # If not, create a monolingual dataset and train them
    else:
        print(f'Creating source tokenizer...')
        source_tokenizer = create_tokenizer('source')
        source_tokenizer.to_txt(source_tokenizer_path)

    if os.path.exists(target_tokenizer_path):
        print(f'Loading target tokenizer...')
        target_tokenizer = WordLevelTokenizer.load(target_tokenizer_path, driver='txt')
    else:
        print(f'Creating target tokenizer...')
        target_tokenizer = create_tokenizer('target')
        target_tokenizer.to_txt(target_tokenizer_path)

    print(f'Source tokenizer vocabulary size: {source_tokenizer.vocab_size}')
    print(f'Target tokenizer vocabulary size: {target_tokenizer.vocab_size}')
    print(f'-' * 100)

    # Initialize the model
    transformer = create_transformer(
        source_tokenizer.vocab_size,
        target_tokenizer.vocab_size,
        config.MAX_SEQ_LEN,
        config.MAX_SEQ_LEN
    )
    print('Model created')

    # Optional: restore weights

    # Use the model to make inference
    input_text = 'We had been wandering, indeed, in the leafless shrubbery an hour in the morning;'
    tokenized_input = source_tokenizer.get_encoder_input(input_text, config.MAX_SEQ_LEN)
    max_output_len = 30

    print(f'Input sentence          : {input_text}')
    print(f'Tokenized input         : {tokenized_input}')
    print(f'Tokenized input shape   : {tokenized_input.shape}')
    print(f'Max output len          : {max_output_len}')
    print(f'-' * 100)

    model_output = transformer.translate(
        input_text,
        source_tokenizer,
        target_tokenizer,
        max_output_len,
    )
    decoded_output = target_tokenizer.decode(model_output)

    print(f'Model output   : {model_output}')
    print(f'Decoded output : {decoded_output}')
