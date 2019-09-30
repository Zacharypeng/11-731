
# Report for 11-731 Assignment 1
by zeyupeng@cs.cmu.edu, [Github Repo](https://github.com/Zacharypeng/11-731.git)

`nmt.py` it's a program to translate German to English. 

## Architecure of the model

### Overall description

The model used the encoder and decoder architecture, with an attention layer between the encoder and the decoder.

### Encoder

The encoder's input is a list of list of training sentences. Each sentence will get go though an word_embedding layer of embeding size 256. And this embeddings will be passed into a layer of bi-direction LSTM. And the encoder will return the LSTM's output, along with the hidden state and cell state. 

### Decoder

The decoder's input is a list of list of target sentences, and the output, hidden state, cell state from the encoder. The target sentences will go through a word_embedding layer of embedding size 256. And for each input batch, the decoder iterates on the sentences on each time step. Each word on each time step, will be the put into a layer of uni-direction LSTM. One thing also need to be mentioned is that, the LSTM's initial state is the hidden state and cell state passed from the encoder. And the output hidden states will again be the input state for the decoder's LSTM in the next time step. Also, for each iteration, the decoder uses teacher forcing, where the input could be the last generated value. The decoder's hidden state combines with the context generated from the attention model, and passes though a linear layer, will be the output score. 

### Attention

For this model, it uses the Global Attention described in [Luong et al., 2015](https://arxiv.org/pdf/1508.04025.pdf) and the DotProduct of target states and source states to calculate score. 

### Decode and Test

When Decoding and test, the model uses greedy search, which means that it will choose the word with the highest probability for each time step. 

## Result and Analysis

For the given validation and test dataset, this model reaches **BLEU score of 25.91** on validation dataset, and **BLEU score of 24.47** on the test dataset. The output result stores in the dev.txt and the test.txt. The training log is in the training_log.txt. Definitely should try Beam Search for future experiments. 

## Experiments

### Decoder LSTM initial state

If the Decoder LSTM's initial state were set to zeros, the model will stuck around training loss of 90 and barly improved from there. But after passing the output hidden state from encoder's LSTM to the decoder LSTM initial state, the performance for this model increased greatly. 

### Teacher forcing

Adding teacher forcing in the decoder while training, improved the performance of the model. The validation perplexity of the model using teacher forcing went below 16. Whereas it stayed above 18 without using teacher forcing. This model was trained with the teacher forcing rate of 0.1, where 10% of the training the LSTM get the last generated value as input and 90% of the training the LSTM get the ground truth as input. 

### MLP vs Linear

For generating key and value from the encoder output, for using attention mechanism, I tried using a multilayer perceptron with activation and also just a linear layer without activation. The performance of just using a linear layer was much better than MLP even at the first 3 training epoch. The reason behind this could be that, I used ReLU as the activation function, so some of the gradient got lost when performing backward prop. 

### LSTM vs LSTM Cell

In the paper, the model used one layer of LSTM cell in decoder. However, I got stuck when implementing the LSTM cell in the first place and switch to use LSTM. And I don't think this change will have any influence on the performance of the model. However, it turned out that I have to deal with difference of dimensions even more when using LSTM. Because it only accepts three dimensional input. 

## Hyperparameters

I used the batch size of 256, learning rate of 1e-3 for training. Embedding size for both encoder and decoder are 256. Hidden size for both encoder LSTM and decoder LSTM are 256. Tearcher forcing rate was set to 0.1. 

## Basic Usage

For taining
```bash
python nmt.py train --train-src=data/train.de-en.de.wmixerprep \
                    --train-tgt=data/train.de-en.en.wmixerprep \ 
                    --dev-src=data/valid.de-en.de.wmixerprep \ 
                    --dev-tgt=data/valid.de-en.en.wmixerprep \ 
                    --vocab=data/vocab.bin \ 
                    --lr=1e-3 --lr-decay=0.5 --batch-size=256 \ 
                    --save-to='model.pt' --valid-niter=800 \
                    --patience=2 --max-num-trial=5
```

For testing
```bash
python nmt.py decode model_2.pt data/test.de-en.de.wmixerprep \
                                data/test.de-en.en.wmixerprep \ 
                                test_output.txt
```

## References
* [Luong et al., 2015](https://arxiv.org/pdf/1508.04025.pdf)

* Idea from my assignment for 11-785 Deep Learning, where I used Pyramid Bi-directinoal LSTM for speech recognition

