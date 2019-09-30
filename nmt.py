# coding=utf-8

"""
A very basic implementation of neural machine translation

python nmt.py train --train-src=data/train.de-en.de.wmixerprep --train-tgt=data/train.de-en.en.wmixerprep --dev-src=data/valid.de-en.de.wmixerprep --dev-tgt=data/valid.de-en.en.wmixerprep --vocab=data/vocab.bin --lr=5e-4 --lr-decay=0.5 --batch-size=128 --save-to='model.pt'

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Set, Union, Any
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter, input_transpose
from vocab import Vocab, VocabEntry
import pdb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
pad_token = '<pad>'


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        self.context_length = hidden_size
        self.query_size = hidden_size

        self.encoder_embedding = nn.Embedding(len(vocab.src), embed_size)
        self.decoder_embedding = nn.Embedding(len(vocab.tgt), embed_size)
        self.encoder_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.decode_lstm = nn.LSTM(input_size=embed_size+hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.attention = Attention()
        self.criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)

        self.bi2uni_mlp = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.key_mlp = nn.Linear(self.hidden_size*2, hidden_size)
        self.value_mlp = nn.Linear(self.hidden_size*2, hidden_size)
        self.query_mlp = nn.Linear(self.hidden_size, self.query_size)
        self.output_mlp = nn.Linear((self.query_size+self.context_length), len(self.vocab.tgt))

    def forward(self, src_sents: List[List[str]], tgt_sents: List[List[str]], teacher_forcing: True):
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`
            teacher_forcing: boolean

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        encoder_output, decoder_init_state, encoder_length_list = self.encode(src_sents)  # B x L x E
        attention_mask = torch.zeros((encoder_output.shape[1], encoder_output.shape[0])).to(DEVICE)  # B x L
        for i in range(attention_mask.shape[0]):
            for j in range(encoder_length_list[i]):
                attention_mask[i][j] = 1
        pred, padded_labels = self.decode(encoder_output, decoder_init_state, attention_mask, tgt_sents, teacher_forcing=True)
        loss_target = rnn.pad_sequence(padded_labels, padding_value=0, batch_first=True)
        loss = self.criterion(pred, loss_target[:, 1:])
        return loss

    def encode(self, src_sents: List[List[str]]):
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable 
                with shape (batch_size, source_sentence_length, encoding_dim), or in other formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """

        input_src = [torch.tensor(sentence) for sentence in self.vocab.src.words2indices(src_sents)]
        input_length = [len(l) for l in input_src]
        padded_input_src = rnn.pad_sequence(input_src)  # L x B x E
        padded_input_src = padded_input_src.to(DEVICE)
        src_encodings = self.encoder_embedding(padded_input_src)  # L x B x E
        # pdb.set_trace()
        src_encodings = rnn.pack_padded_sequence(src_encodings, torch.tensor(input_length), batch_first=False)
        # pdb.set_trace()
        src_encodings, (h_n, c_n) = self.encoder_lstm(src_encodings)
        # pdb.set_trace()
        c_n = self.bi2uni_mlp(torch.cat([c_n[0], c_n[1]], dim=1))
        # pdb.set_trace()
        h_n = torch.tanh(c_n)
        output, length_list = rnn.pad_packed_sequence(src_encodings)
        return output, (h_n, c_n), length_list

    def decode(self, encoder_output: torch.Tensor, decoder_init_state, attention_mask: Any, tgt_sents: List[List[str]], teacher_forcing = True):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            encoder_output: B x L x E
            attention_mask: B x L
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        target_input = [torch.tensor(sentence) for sentence in self.vocab.tgt.words2indices(tgt_sents)]
        padded_input = rnn.pad_sequence(target_input, batch_first=True)
        padded_input = padded_input.to(DEVICE)
        embeddings = self.decoder_embedding(padded_input)  # B x L x E
        context = torch.zeros(embeddings.size(0), self.context_length)  # B x C (256)
        batch_size = embeddings.size(0)
        context = context.unsqueeze(1).to(DEVICE)  # B x 1 x C
        teacher_forcing_rate = 0.1
        output = []
        for t in range(embeddings.size(1) - 1):
            if teacher_forcing:
                if t == 0 or np.random.random() >= teacher_forcing_rate:
                    # pdb.set_trace()  # context: B x 1 x C
                    input = torch.cat((embeddings[:, t, :].unsqueeze(1), context), dim=2)  # B x 1 x (E + C)
                else:
                    # use teacher forcing
                    pred_char = torch.argmax(score.squeeze(1), dim=1)  # B x 1
                    pred_char = pred_char.reshape(batch_size, 1)
                    temp_char = self.decoder_embedding(pred_char)  # B x 1 x 256
                    input = torch.cat((temp_char, context), dim=2)
            else:
                input = torch.cat((embeddings[:, t, :].unsqueeze(1), context), dim=2)  # B x 1 x (E + C)
            input = input.permute(1, 0, 2)
            if t == 0:
                decoder_init_state = (decoder_init_state[0].unsqueeze(0), decoder_init_state[1].unsqueeze(0))
            # pdb.set_trace()
            out, decoder_init_state = self.decode_lstm(input, decoder_init_state)

            # query and hidden could have different size
            key = self.key_mlp(encoder_output)
            value = self.value_mlp(encoder_output)
            # query = self.query_mlp(hidden)
            query = decoder_init_state[0]  # B in the middle
            context = self.attention(query, key, value, attention_mask)  # B x 1 x L
            # pdb.set_trace()
            pred_input = torch.cat((query.permute(1, 0, 2), context), 2)
            score = self.output_mlp(pred_input)
            output.append(score)
        prediction = torch.stack(output, dim=1).squeeze(2)
        prediction = prediction.permute(0, 2, 1)
        return prediction, padded_input

    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70):
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        hypothesis = [torch.tensor(1).numpy()]
        encoder_output, decoder_init_state, encoder_length_list = self.encode([src_sent])  # B x L x E
        attention_mask = torch.zeros((encoder_output.shape[1], encoder_output.shape[0])).to(DEVICE)  # B x L
        for i in range(attention_mask.shape[0]):
            for j in range(encoder_length_list[i]):
                attention_mask[i][j] = 1
        key = self.key_mlp(encoder_output)
        value = self.value_mlp(encoder_output)
        context = torch.zeros((1, self.context_length))  # B x C (256)
        context = context.to(DEVICE)
        pred_char = torch.tensor([1]).to(DEVICE)
        for t in range(max_decoding_time_step):
            if hypothesis[-1] == 2:
                break
            # pdb.set_trace()
            embed_char = self.decoder_embedding(pred_char).unsqueeze(0)
            if t == 0:
                test_input = torch.cat((embed_char, context.unsqueeze(0)), dim=2)
            else:
                test_input = torch.cat((embed_char, context), dim=2)
            if t == 0:
                decoder_init_state = (decoder_init_state[0].unsqueeze(0), decoder_init_state[1].unsqueeze(0))
            # test_input = test_input.unsqueeze(0)
            # pdb.set_trace()
            out, decoder_init_state = self.decode_lstm(test_input, decoder_init_state)
            # pdb.set_trace()
            query = decoder_init_state[0]  # B in the middle
            context = self.attention(query, key, value, attention_mask)  # B x 1 x L
            pred_input = torch.cat((query.permute(1, 0, 2), context), 2)
            score = self.output_mlp(pred_input)
            pred_char = torch.argmax(score.squeeze(1), dim=1)  # B x 1
            # pred_char = pred_char.reshape(batch_size, 1)
            # temp_char = self.decoder_embedding(pred_char)  # B x 1 x 256
            hypothesis.append(pred_char.cpu().numpy())

        translate = [self.vocab.tgt.id2word[char[0]] for char in hypothesis[1:-1]]
        # pdb.set_trace()
        return translate

    def evaluate_ppl(self, dev_data: List[Any], batch_size: int = 32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`
        with torch.no_grad():
            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                loss = self.forward(src_sents, tgt_sents, teacher_forcing=False).sum()

                cum_loss += loss.item()
                tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

            ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        raise NotImplemented
        # return model

    def save(self, path: str):
        """
        Save current model to file
        """

        torch.save


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask):
        """
            query: 1 x B x H
            key: L x B x 128
            value: L/8 x B x 128
        """

        key = key.permute(1, 2, 0)  # B x H x L
        query = query.permute(1, 0, 2)  # B x 1 x H
        # pdb.set_trace()
        energy = torch.bmm(query, key)  # B x 1 x L
        # pdb.set_trace()
        attention = F.softmax(energy, dim=2)  # B x 1 x L
        attention = attention * mask.unsqueeze(1)  # B x 1 x L
        norm_attention = F.normalize(attention, p=1, dim=2)  # B x 1 x L
        # pdb.set_trace()
        trans_value = value.permute(1, 0, 2)  # B x L x 128
        context = torch.bmm(norm_attention, trans_value)  # B x 1 x H
        # context = context.permute(1, 0, 2)  # 1 x B x H
        return context


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    # bleu_score = corpus_bleu([[ref] for ref in references],
    #                          [hyp.value for hyp in hypotheses])
    # pdb.set_trace()
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    lr_decay = float(args['--lr-decay'])
    learning_rate = float(args['--lr'])

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    HIDDEN_SIZE = 256
    EMBEDDING_SIZE = 256
    DROP_OUT = 0.2

    model = NMT(embed_size=EMBEDDING_SIZE,
                hidden_size=HIDDEN_SIZE,
                dropout_rate=float(DROP_OUT),
                vocab=vocab)
    model = model.to(DEVICE)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            batch_size = len(src_sents)

            optim.zero_grad()

            loss = model(src_sents, tgt_sents, False)
            report_loss += loss.item()
            cum_loss += loss.item()

            loss.backward()
            optim.step()
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    torch.save(model, model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model = torch.load(model_save_path)
                        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
        # pdb.set_trace()
        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    # model = NMT.load(args['MODEL_PATH'])
    model = torch.load(args['MODEL_PATH'])

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        # top_hypotheses = [hyps[0] for hyps in hypotheses]
        top_hypotheses = hypotheses
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
