# Bidirectional Encoder Representations from Transformers (BERT)

Pytorch lightning implementation of the BERT encoder-only architecture 
as described in the [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) paper.
Along with the architecture, the repo contains the code to run training and inference on
a next POI recommendation task. A Tokenizer and a Dataloader are provided as well. 
The dataloader uses the [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) dataset.