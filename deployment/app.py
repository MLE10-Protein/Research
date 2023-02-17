import gradio as gr
import os
import gc
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, LSTM, CuDNNLSTM, GlobalMaxPooling1D

import tensorflow as tf
from huggingface_hub import hf_hub_url, cached_download


class Sequence:
    codes = {c: i+1 for i, c in enumerate(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])}

    @classmethod
    def integer_encoding(cls, data):
        """
        - Encodes code sequence to integer values.
        - 20 common amino acids are taken into consideration
        and remaining four are categorized as 0.
        """
        return np.array([cls.codes.get(code, 0) for code in data])

    @classmethod
    def prepare(cls, sequence):
        sequence = sequence.strip().upper()
        ie = cls.integer_encoding(sequence)
        max_length = 100
        padded_ie = pad_sequences([ie], maxlen=max_length, padding='post', truncating='post')
        all_ohe = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] + [0]*(100-21))
        return padded_ie, to_categorical(np.array([padded_ie[0], all_ohe]))[:1]


def residual_block(data, filters, d_rate):
    """
    _data: input
    _filters: convolution filters
    _d_rate: dilation rate
    """
    
    shortcut = data
    
    bn1 = BatchNormalization()(data)
    act1 = Activation('relu')(bn1)
    conv1 = Conv1D(filters, 1, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)
    
    #bottleneck convolution
    bn2 = BatchNormalization()(conv1)
    act2 = Activation('relu')(bn2)
    conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(act2)
    
    #skip connection
    x = Add()([conv2, shortcut])
    
    return x

def get_model():
    # model
    x_input = Input(shape=(100, 21))
    
    #initial conv
    conv = Conv1D(128, 1, padding='same')(x_input) 
    
    # per-residue representation
    res1 = residual_block(conv, 128, 2)
    res2 = residual_block(res1, 128, 3)
    
    x = MaxPooling1D(3)(res2)
    x = Dropout(0.5)(x)
    
    # softmax classifier
    x = Flatten()(x)
    x_output = Dense(1000, activation='softmax', kernel_regularizer=l2(0.0001))(x)
    
    model2 = Model(inputs=x_input, outputs=x_output)
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    weights = cached_download(hf_hub_url("jonathang/Protein_Family_Models", 'model2.h5'))
    model2.load_weights(weights)

    return model2

def get_lstm_model():
    x_input = Input(shape=(100,))
    emb = Embedding(21, 128, input_length=100)(x_input)
    bi_rnn = Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(emb)
    # bi_rnn = CuDNNLSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))(emb)
    x = Dropout(0.3)(bi_rnn)

    # softmax classifier
    x_output = Dense(1000, activation='softmax')(x)

    model1 = Model(inputs=x_input, outputs=x_output)
    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    weights = cached_download(hf_hub_url("jonathang/Protein_Family_Models", 'model1.h5'))
    model1.load_weights(weights)
    return model1


cnn_model = get_model()
lstm_model = get_lstm_model()

mappings_path = cached_download(hf_hub_url("jonathang/Protein_Family_Models", 'prot_mappings.json'))
with open(mappings_path) as f:
    prot_mappings = json.load(f)

def greet(Amino_Acid_Sequence):
    padded_seq, processed_seq = Sequence.prepare(Amino_Acid_Sequence)
    cnn_raw_prediction = cnn_model.predict(processed_seq)[0]
    lstm_raw_prediction = lstm_model.predict(padded_seq)[0]
    joined_prediction = cnn_raw_prediction*0.7 + lstm_raw_prediction*0.3
    cnn_idx = cnn_raw_prediction.argmax()
    lstm_idx = lstm_raw_prediction.argmax()
    idx = joined_prediction.argmax()
    cnn_fam_asc = prot_mappings['id2fam_asc'][str(cnn_idx)]
    cnn_fam_id = prot_mappings['fam_asc2fam_id'][cnn_fam_asc]
    lstm_fam_asc = prot_mappings['id2fam_asc'][str(lstm_idx)]
    lstm_fam_id = prot_mappings['fam_asc2fam_id'][lstm_fam_asc]
    fam_asc = prot_mappings['id2fam_asc'][str(idx)]
    fam_id = prot_mappings['fam_asc2fam_id'][fam_asc]
    joined_probs = {prot_mappings['id2fam_asc'][str(i)] + ' ' + prot_mappings['fam_asc2fam_id'][prot_mappings['id2fam_asc'][str(i)]]: float(joined_prediction[i]) for i in range(len(joined_prediction))}
    cnn_probs = {prot_mappings['id2fam_asc'][str(i)] + ' ' + prot_mappings['fam_asc2fam_id'][prot_mappings['id2fam_asc'][str(i)]]: float(cnn_raw_prediction[i]) for i in range(len(cnn_raw_prediction))}
    lstm_probs = {prot_mappings['id2fam_asc'][str(i)] + ' ' + prot_mappings['fam_asc2fam_id'][prot_mappings['id2fam_asc'][str(i)]]: float(lstm_raw_prediction[i]) for i in range(len(lstm_raw_prediction))}
    gc.collect()
    return joined_probs, cnn_probs, lstm_probs, f"""
Input is {Amino_Acid_Sequence}.
Processed input is:
{processed_seq}

CNN says: Family Accession={cnn_fam_asc} and ID={cnn_fam_id}
LSTM says: Family Accession={lstm_fam_asc} and ID={lstm_fam_id}

0.7 * cnn and 0.3 * lstm ensemble model makes prediction which maps to:
Family Accession={fam_asc} and ID={fam_id}

Raw Joined Prediction:
{joined_prediction}
"""

iface = gr.Interface(fn=greet, inputs="text", outputs=[gr.Label(num_top_classes=5, label="Ensemble Family Predictions"), gr.Label(num_top_classes=5, label="CNN Family Predictions"), gr.Label(num_top_classes=5, label="LSTM Family Predictions"), "text"])
iface.launch()
