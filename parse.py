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



# In[2]:


from music21 import converter, instrument, note, chord, stream

def get_notes():
    songs = glob('midi/*.mid')
    notes = []
    for file in songs:
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
            #print(mlen)
            if(mlen == 0):
                continue
            try:
                length,istrue = note.duration.quarterLengthToClosestType(mlen)
            except:
                continue
            lengthstr = str((1/note.duration.convertTypeToNumber(length))*4)
            if(float(lengthstr) > 4):
                lengthstr = str(4)
            elif(float(lengthstr) < 0.25):
                lengthstr = str(0.25)
            lengthstr = 'L'+lengthstr
            if isinstance(element, note.Note):
                # if element is a note, extract pitch
                notes.append(str(element.pitch)+lengthstr)
            if isinstance(element, note.Rest):
                notes.append(str('R') + lengthstr)
            elif(isinstance(element, chord.Chord)):
                # if element is a chord, append the normal form of the
                # chord (a list of integers) to the list of notes.
                notes.append('.'.join(str(n) for n in element.normalOrder) + lengthstr)
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

get_notes()
