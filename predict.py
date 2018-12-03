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

def midi2net(notes, n_vocab):
    sequence_length = 100

    # Extract the unique pitches in the list of notes.
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    out = []
    # create input sequences and the corresponding outputs
    for i in range(len(notes)-sequence_length, len(notes)):
        out.append(note_to_int[notes[i]])

    return out


def generate():
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))
    network_input = network.get_inputSequences(notes, pitchnames, n_vocab)
    normalized_input = np.array(network_input)
    normalized_input = np.reshape(normalized_input, (len(network_input), 100, 1))
    model = network.create_network(normalized_input, n_vocab)

    songs = glob('predict/*.mid')

    for file in songs:
        notes = []
        # converting .mid file to stream object
        print('parsing file' + file)
        midi = converter.parse(file)
        notes_to_parse = []
        try:
            # Given a single stream, partition into a part for each unique instrument
            parts = instrument.partitionByInstrument(midi)
        except:
            pass
        if parts: # if parts has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            mlen = element.duration.quarterLength
            if(mlen == 0):
                continue
            try:
                length,istrue = note.duration.quarterLengthToClosestType(mlen)
            except:
                continue
            #Standardize the notes to be 32/16/8/4/2/1
            lengthstr = str((1/note.duration.convertTypeToNumber(length))*4)
            if(float(lengthstr) > 4):
                lengthstr = str(4)
            elif(float(lengthstr) < 0.25):
                lengthstr = str(0.25)
            lengthstr = 'L'+lengthstr
            if isinstance(element, note.Note):
                # If its a note wright the note
                notes.append(str(element.pitch)+lengthstr)
            if isinstance(element, note.Rest):
                # If its a rest wright it
                notes.append(str('R') + lengthstr)
            elif(isinstance(element, chord.Chord)):
                # If its a chord wright it
                notes.append('.'.join(str(n) for n in element.normalOrder) + lengthstr)

        seq = midi2net(notes,n_vocab)
        prediction_output = generate_notes(model, network_input, pitchnames, n_vocab,seq)
        create_midi(prediction_output,file + '.predicted')




def generate_notes(model, network_input, pitchnames, n_vocab,pattern):


    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    prediction_output = []
    for note_index in range(100):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input)

        # Predicted output is the argmax(P(h|D))
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        # Next input to the model
        pattern = pattern[1:len(pattern)]

    print('Notes Generated...')
    return prediction_output

def create_midi(prediction_output,file):
    offset = 0
    output_notes = []
    length = 1
    for pattern in prediction_output:

        code,strlength = pattern.split('L')
        try:
            length = float(strlength)
        except:
            length = 1
        if('R' in code):
        # Check for rest
            str  = pattern.replace('R','')
            print(str)
            offset += length
        elif ('.' in code) or code.isdigit():
        # Check for chord
            notes_in_chord = code.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.duration.quarterLength = length
            new_chord.offset = offset
            output_notes.append(new_chord)
            offset += length

        else:
        # Otherwise is note
            new_note = note.Note(code)
            new_note.duration.quarterLength = length
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            offset += length
    midi_stream = stream.Stream(output_notes)
    print('Saving Output file as midi....')
    midi_stream.write('midi', fp=file)

generate()
