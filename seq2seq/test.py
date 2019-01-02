# -*- coding: utf-8 -*-
__author__ = 'liyang54'
from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import time

# seq2seq model

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# encode²ãÊä³öœá¹û
def get_encoder_layer(input_data, rnn_size, num_layers,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):

    '''
    ¹¹ÔìEncoder²ã
    ²ÎÊýËµÃ÷£º
    - input_data: ÊäÈëtensor
    - rnn_size: rnnÒþ²ãœáµãÊýÁ¿
    - num_layers: ¶ÑµþµÄrnn cellÊýÁ¿
    - source_sequence_length: ÔŽÊýŸÝµÄÐòÁÐ³€¶È
    - source_vocab_size: ÔŽÊýŸÝµÄŽÊµäŽóÐ¡
    - encoding_embedding_size: embeddingµÄŽóÐ¡
    '''
    # Encoder embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    # RNN cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell
        # lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
        #encoder_state±íÊŸencoder×îÖÕ×ŽÌ¬
        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=source_sequence_length, dtype=tf.float32)

    # lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(encoding_embedding_size, forget_bias=1.0)
    # input_batch_size = tf.shape(input_data)[0] + 0
    # initial_state_sentence = lstm_cell_fw.zero_state(input_batch_size, dtype=tf.float32)
    # encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cell_fw, encoder_embed_input, dtype=tf.float32,
    #                                              initial_state=initial_state_sentence)


        return encoder_output, encoder_state



def process_decoder_input(data, vocab_to_int, batch_size):
    '''
    ²¹³ä<GO>£¬²¢ÒÆ³ý×îºóÒ»žö×Ö·û
    '''
    # cutµô×îºóÒ»žö×Ö·û
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return decoder_input


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input, batch_size):
    '''
    ¹¹ÔìDecoder²ã
    ²ÎÊý£º
    - target_letter_to_int: targetÊýŸÝµÄÓ³Éä±í
    - decoding_embedding_size: embedÏòÁ¿ŽóÐ¡
    - num_layers: ¶ÑµþµÄRNNµ¥ÔªÊýÁ¿
    - rnn_size: RNNµ¥ÔªµÄÒþ²ãœáµãÊýÁ¿
    - target_sequence_length: targetÊýŸÝÐòÁÐ³€¶È
    - max_target_sequence_length: targetÊýŸÝÐòÁÐ×îŽó³€¶È
    - encoder_state: encoder¶Ë±àÂëµÄ×ŽÌ¬ÏòÁ¿
    - decoder_input: decoder¶ËÊäÈë
    '''
    # 1. Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 2. ¹¹ÔìDecoderÖÐµÄRNNµ¥Ôª
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell
    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    # 3. OutputÈ«Á¬œÓ²ã
    output_layer = Dense(target_vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))


    # 4. Training decoder
    # TrainingHelperÓÃÓÚtrainœ×¶Î
    # encoder_state±íÊŸencoderµÄfinal state
    with tf.variable_scope("decode"):
        # µÃµœhelp¶ÔÏó
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        # ¹¹Ôìdecoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)
    # 5. Predicting decoder
    # Óëtraining¹²Ïí²ÎÊý
    # GreedyEmbeddingHelperÓÃÓÚtestœ×¶Î
    with tf.variable_scope("decode", reuse=True):
        # ŽŽœšÒ»žö³£Á¿tensor²¢žŽÖÆÎªbatch_sizeµÄŽóÐ¡
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                start_tokens,
                                                                target_letter_to_int['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                        predicting_helper,
                                                        encoder_state,
                                                        output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                            impute_finished=True,
                                                            maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output



def seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoding_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers, target_letter_to_int, batch_size, decoding_embedding_size):

    # »ñÈ¡encoderµÄ×ŽÌ¬Êä³ö
    _, encoder_state = get_encoder_layer(input_data,
                                  rnn_size,
                                  num_layers,
                                  source_sequence_length,
                                  source_vocab_size,
                                  encoding_embedding_size)


    # Ô€ŽŠÀíºóµÄdecoderÊäÈë
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)

    # œ«×ŽÌ¬ÏòÁ¿ÓëÊäÈëŽ«µÝžødecoder
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                       decoding_embedding_size,
                                                                       num_layers,
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       encoder_state,
                                                                       decoder_input,
                                                                       batch_size)

    return training_decoder_output, predicting_decoder_output