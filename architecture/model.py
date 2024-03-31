from architecture.blocks.multihead_attention import MultiHeadAttentionBlock
from architecture.blocks.bert_embeddings import BERTEmbeddings
from architecture.blocks.feed_forward import FeedForwardBlock
from architecture.blocks.projection import MLMProjectionLayer
from architecture.blocks.projection import NSPProjectionLayer
from architecture.blocks.encoder_block import EncoderBlock
from architecture.blocks.encoder import Encoder
import pytorch_lightning as pl
import torch.nn as nn
import torch


class BERT(nn.Module):

    def __init__(self,
                 embedding_layer: BERTEmbeddings,
                 encoder: Encoder,
                 mlm_projection_layer: MLMProjectionLayer,
                 nsp_projection_layer: NSPProjectionLayer
                 ) -> None:

        super().__init__()

        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.mlm_projection_layer = mlm_projection_layer
        self.nsp_projection_layer = nsp_projection_layer

    def forward(self, x, attention_mask, segment_info):
        x = self.embedding_layer(x, segment_info)
        x = self.encoder(x, attention_mask)
        return self.mlm_projection_layer(x), self.nsp_projection_layer(x)

    @classmethod
    def build(cls,
              vocab_size: int,
              seq_len: int,
              embed_dim: int,
              num_encoders: int,
              heads: int,
              dropout: float,
              d_ff: int
              ):
        """
        Build a bert module
        """

        # Create the embedding layer
        embedding_layer = BERTEmbeddings(vocab_size, embed_dim, seq_len, dropout)

        # Create a list of encoder blocks
        encoder_blocks = []
        for _ in range(num_encoders):

            encoder_self_attention_block = MultiHeadAttentionBlock(embed_dim, heads, dropout)
            feed_forward_block = FeedForwardBlock(embed_dim, d_ff, dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        # Create the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
        encoder = Encoder(nn.ModuleList(encoder_blocks))

        # Create the projection layer
        mlm_projection_layer = MLMProjectionLayer(embed_dim, vocab_size)
        nsp_projection_layer = NSPProjectionLayer(embed_dim)

        # Create the transformer
        bert = BERT(
            embedding_layer,
            encoder,
            mlm_projection_layer,
            nsp_projection_layer
        )

        # Initialize the model parameters to train faster (otherwise they are initialized with
        # random values). We use Xavier initialization
        for p in bert.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return bert


class LightningBERT(pl.LightningModule):

    def __init__(self, model, criterion, optimizer, scheduler=None):
        super(LightningBERT, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def common_step(self, batch, batch_idx):

        is_next = batch['is_next']
        bert_input = batch['bert_input']
        bert_label = batch['bert_label']
        attention_mask = batch['attention_mask']
        segment_label = batch['segment_label']

        # Run the input through the model
        mlm_out, nsp_out = self.model(bert_input, attention_mask, segment_label)

        # Compute the two losses
        mlm_loss = self.criterion(mlm_out.transpose(1, 2), bert_label)  # Transpose to (batch, vocab_size, seq_len)
        nsp_loss = self.criterion(nsp_out, torch.flatten(is_next))

        return mlm_loss + nsp_loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        if self.scheduler:
            return {'optimizer': optimizer, 'lr_scheduler': self.scheduler}
        else:
            return optimizer


