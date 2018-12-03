import sys
import re
import numpy as np
import pandas as pd
import music21
from glob import glob
import IPython
from tqdm import tqdm
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from music21 import converter, instrument, note, chord, stream
import network

def prepare_sequences(notes, n_vocab):
    sequence_length = 100

    # Extract the unique pitches in the list of notes.
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format comatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input
    network_input = network_input / float(n_vocab)

    # one hot encode the output vectors
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def train(model, network_input, network_output, epochs):
    # Create checkpoint to save the best model weights.
    filepath = 'weights.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True)
    model.fit(network_input, network_output, epochs=epochs, batch_size=150, callbacks=[checkpoint],verbose=2)


def train_network():
    epochs = 50
    notes = []
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
    vocablen = len(set(notes))

    net_in, net_out = prepare_sequences(notes,vocablen)

    model = network.create_network(net_in, vocablen)

    train(model, net_in, net_out, epochs)

train_network()
