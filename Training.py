import os as os
import json as js
import numpy as np
import random as ran
import tensorflow as tf
import keras.utils as ut
import keras.models as mod
import keras.layers as lay
import keras.losses as los
import keras.metrics as kmet
import keras.optimizers as opt
import keras.activations as act
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import tokenizers.implementations as imp
from keras.layers import Dropout
def AddStartEndOOV(Tokens:list[list[str]],
                   Indexes:list[list[int]],
                   Vocabulary:dict,
                   Characters:dict) -> tuple[list[list[str]],
                                             list[list[int]],
                                             dict]:
    MaxIndex1 = max(list(Vocabulary.values()))          
    MaxIndex2 = max(list(Characters.values()))
    StartIndex1 = MaxIndex1 + 1                        
    EndIndex1 = MaxIndex1 + 2
    OOVIndex1 = MaxIndex1 + 3
    PaddingIndex1 = MaxIndex1 + 4
    StartIndex2 = MaxIndex2 + 1
    EndIndex2 = MaxIndex2 + 2
    OOVIndex2 = MaxIndex2 + 3
    PaddingIndex2 = MaxIndex2 + 4
    Vocabulary['<START>'] = StartIndex1                
    Vocabulary['<END>'] = EndIndex1
    Vocabulary['<OOV>'] = OOVIndex1
    Vocabulary['<PADDING>'] = PaddingIndex1
    Characters['<START>'] = StartIndex2
    Characters['<END>'] = EndIndex2
    Characters['<OOV>'] = OOVIndex2
    Characters['<PADDING>'] = PaddingIndex2
    for i, (Token, Index) in enumerate(zip(Tokens, Indexes)):           
        Tokens[i] = ['<START>'] + Token + ['<END>']
        Indexes[i] = [StartIndex1] + Index + [EndIndex1]
    return Tokens, Indexes, Vocabulary, Characters

def Fetch(Path:str,
          sVocabulary:int) -> tuple[list[list[str]],
                                    list[list[int]],
                                    dict,
                                    dict]:
    if os.path.exists('Tokens.json') and os.path.exists('Indexes.json') and os.path.exists('Vocabulary.json'):
        with open('Tokens.json', mode='r', encoding='UTF-8') as F:
            Tokens = js.load(F)
        with open('Indexes.json', mode='r', encoding='UTF-8') as F:
            Indexes = js.load(F)
        with open('Vocabulary.json', mode='r', encoding='UTF-8') as F:
            Vocabulary = js.load(F)
        with open('Characters.json', mode='r', encoding='UTF-8') as F:
            Characters = js.load(F)
    else:
        with open(file=Path, mode='r', encoding='UTF-8',errors='ignore') as F:
            S = F.read()
        Passwords = S.split('\n')#[:100000]
        Characters = {}
        for Password in Passwords:
            for Character in Password:
                if Character not in Characters:
                    Characters[Character] = len(Characters)         
        Tokenizer = imp.ByteLevelBPETokenizer()
        Tokenizer.train_from_iterator(iterator=Passwords,           
                                      vocab_size=sVocabulary,
                                      min_frequency=10)
        Tokens = []
        Indexes = []
        Vocabulary = Tokenizer.get_vocab()                         
        for Password in Passwords:
            Encoded = Tokenizer.encode(Password)
            Tokens.append(Encoded.tokens)                           
            Indexes.append(Encoded.ids)                             
        Tokens, Indexes, Vocabulary, Characters = AddStartEndOOV(Tokens, Indexes, Vocabulary, Characters)
        with open('Tokens.json', mode='w', encoding='UTF-8') as F:
            js.dump(Tokens, F)
        with open('Indexes.json', mode='w', encoding='UTF-8') as F:
            js.dump(Indexes, F)
        with open('Vocabulary.json', mode='w', encoding='UTF-8') as F:
            js.dump(Vocabulary, F)
        with open('Characters.json', mode='w', encoding='UTF-8') as F:
            js.dump(Characters, F)
    return Tokens, Indexes, Vocabulary, Characters

def Lag(Tokens:list[list[str]],
        Indexes:list[list[int]],
        nLag:int,
        nHorizon:int,
        Vocabulary:dict,
        Characters:dict,
        MaxLength2:int=None) -> tuple[np.ndarray,
                                  np.ndarray,
                                  np.ndarray,
                                  int]:
    X1 = []
    X2 = []
    Y = []
    for Token, Index in zip(Tokens, Indexes):
        Index = [Vocabulary['<PADDING>'] for _ in range(nLag)] + Index         
        Index2 = [[Characters['<PADDING>']] for _ in range(nLag)]                  
        for t in Token:
            if t in ['<START>', '<END>', '<OOV>']:
                Index2.append([Characters[t]])
            else:
                Index2.append([])           
                for Character in t:            
                    Index2[-1].append(Characters.get(Character, Characters['<OOV>']))   
        Length = len(Index)
        for i in range(Length - nLag - nHorizon + 1):                      
            X1.append(Index[i:i + nLag])
            Y.append(Index[i + nHorizon:i + nLag + nHorizon])
            x2 = []
            for j in Index2[i:i + nLag]:
                x2.extend(j)
            X2.append(x2)
    X1 = np.array(X1, dtype=np.int32)               
    if MaxLength2 is None:
        MaxLength2 = max([len(i) for i in X2])
    X2 = ut.pad_sequences(X2, maxlen=MaxLength2, dtype='int32', padding='pre', value=Characters['<PADDING>'])   
    Y = np.array(Y, dtype=np.int32)
    return X1, X2, Y, MaxLength2

def PlotLoss(History:dict) -> None:
    nEpoch = len(History['loss'])
    Epochs = np.arange(start=1, stop=nEpoch + 1, step=1)
    plt.plot(Epochs, History['loss'], ls='-', lw=1, c='teal', marker='o', ms=3, label='Train')
    plt.plot(Epochs, History['val_loss'], ls='-', lw=1, c='crimson', marker='o', ms=3, label='Validation')
    plt.title('Model Loss Over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('CCE')
    plt.legend()
    plt.show()

def PlotAcc(History:dict) -> None:
    nEpoch = len(History['accuracy'])
    Epochs = np.arange(start=1, stop=nEpoch + 1, step=1)
    plt.plot(Epochs, History['accuracy'], ls='-', lw=1, c='blue', marker='o', ms=3, label='Train')
    plt.plot(Epochs, History['val_accuracy'], ls='-', lw=1, c='red', marker='o', ms=3, label='Validation')
    plt.title('Model Accuracy Over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

def SummarizeModel(Model:mod.Model) -> None:
    print('_' * 60)
    print('Model Summary:')
    Model.summary()
    print('_' * 60)

def PlotModel(Model:mod.Model) -> None:
    ut.plot_model(model=Model, to_file='Model.png', show_shapes=True, expand_nested=True, dpi=384)

def SaveModel(Model:mod.Sequential) -> None:
    mod.save_model(model=Model, filepath='Model', overwrite=True, include_optimizer=True)

def ResetSettings(RandomState:int,
                  Style:str) -> None:
    ran.seed(a=RandomState)
    plt.style.use(style=Style)
    np.random.seed(seed=RandomState)
    tf.random.set_seed(seed=RandomState)
    os.environ['PYTHONHASHSEED'] = str(RandomState)
                    
RandomState = 0
Style = 'ggplot'
Path = 'train_hotmail.txt'
embeddings_file = 'crawl-300d-2M-subword.vec'
sVocabulary0 = 30000
nLag = 5
nHorizon = 1
sTrain = 0.75
sValidation = 0.25
sEmbedding = 300
nLSTM = 32
ReturnSequences = False
nDense = [512,256]
LR = 8e-4
Optimizer = opt.Adam(learning_rate=LR)
sBatch = 128
nEpoch =12

ResetSettings(RandomState, Style)

Tokens, Indexes, Vocabulary, Characters = Fetch(Path, sVocabulary0)
nD = len(Tokens)                          
nDtr = round(sTrain * nD)
nDva = round(sValidation * nD)
trTokens = Tokens[:nDtr]
vaTokens = Tokens[nDtr:]
trIndexes = Indexes[:nDtr]
vaIndexes = Indexes[nDtr:]
nY = len(Vocabulary)                      
print("nY:",nY)

trX1, trX2, trY, MaxLength2 = Lag(trTokens, trIndexes, nLag, nHorizon, Vocabulary, Characters)
vaX1, vaX2, vaY, _ = Lag(vaTokens, vaIndexes, nLag, nHorizon, Vocabulary, Characters, MaxLength2=MaxLength2)

trOHY = ut.to_categorical(trY, num_classes=nY)     
vaOHY = ut.to_categorical(vaY, num_classes=nY)

embeddings_dictionary = {}                         
with open(embeddings_file, mode='r', encoding='UTF-8', newline='\n', errors='ignore') as F:
    next(F)                                                        
    for line in F:
        values = line.rstrip().split(' ')           
        word = values[0]
        coefs = np.asarray(values[1:], dtype=np.float32)                
        embeddings_dictionary[word] = coefs                           

embedding_matrix = np.zeros(shape=(nY, sEmbedding), dtype=np.float32)     
for Token, Index in Vocabulary.items():
    embedding_vector = embeddings_dictionary.get(Token)                     
    if embedding_vector is not None:                                        
        embedding_matrix[Index] = embedding_vector                          

del embeddings_dictionary
embedding_layer1 = lay.Embedding(input_dim=nY, output_dim=sEmbedding, input_length=nLag, weights=[embedding_matrix])            
embedding_layer2 = lay.Embedding(input_dim=len(Characters), output_dim=sEmbedding, input_length=len(trX2[0]))                   
encoder_inputs_placeholder1 = lay.Input(shape=(nLag, ))
encoder_inputs_placeholder2 = lay.Input(shape=(len(trX2[0]), ))
x1 = embedding_layer1(encoder_inputs_placeholder1)                          
x2 = embedding_layer2(encoder_inputs_placeholder2)
encoder1_1 = lay.LSTM(nLSTM, return_state=False, return_sequences=True)
encoder1_2 = lay.LSTM(nLSTM, return_state=True)                             
encoder2_1 = lay.LSTM(nLSTM, return_state=False, return_sequences=True)
encoder2_2 = lay.LSTM(nLSTM, return_state=True)
x1 = encoder1_1(x1)                                                          
x2 = encoder2_1(x2)
_, h1, c1 = encoder1_2(x1)                                                   
_, h2, c2 = encoder2_2(x2)
encoder_states1 = [h1, c1]
encoder_states2 = [h2, c2]
#decoder
decoder_inputs_placeholder1 = lay.Input(shape=(nLag, ))
decoder_inputs_placeholder2 = lay.Input(shape=(len(trX2[0]), ))
decoder_inputs_x1 = embedding_layer1(decoder_inputs_placeholder1)
decoder_inputs_x2 = embedding_layer2(decoder_inputs_placeholder2)
decoder_lstm1 = lay.LSTM(nLSTM, return_sequences=True, return_state=True)
decoder_lstm2 = lay.LSTM(nLSTM, return_sequences=True, return_state=True)
decoder_outputs1, _, _ = decoder_lstm1(decoder_inputs_x1, initial_state=encoder_states1)        
decoder_outputs2, _, _ = decoder_lstm2(decoder_inputs_x2, initial_state=encoder_states2)
decoder_outputs1 = lay.Flatten()(decoder_outputs1)               
decoder_outputs2 = lay.Flatten()(decoder_outputs2)
decoder_outputs1 = lay.MultiHeadAttention(num_heads=128, key_dim=32, attention_axes=(1, ))(decoder_outputs1, decoder_outputs1)    
decoder_outputs2 = lay.MultiHeadAttention(num_heads=128, key_dim=32, attention_axes=(1, ))(decoder_outputs2, decoder_outputs2)    
decoder_outputs = lay.Concatenate()([decoder_outputs1, decoder_outputs2])
decoder_outputs = lay.Dense(nLag * nY, activation=act.linear)(decoder_outputs)       
decoder_outputs = lay.Reshape(target_shape=(nLag, nY))(decoder_outputs)
decoder_outputs = lay.Softmax(axis=-1)(decoder_outputs)                     
Model = mod.Model([encoder_inputs_placeholder1,
                   encoder_inputs_placeholder2,
                   decoder_inputs_placeholder1,
                   decoder_inputs_placeholder2],
                  decoder_outputs)
Model.compile(optimizer=Optimizer, loss=los.CategoricalCrossentropy(),metrics=['accuracy'])
History = Model.fit([trX1, trX2, trX1, trX2],
                    trOHY,
                    batch_size=sBatch,
                    epochs=nEpoch,
                    validation_data=([vaX1, vaX2, vaX1, vaX2], vaOHY),
                    shuffle=True).history                                  
SaveModel(Model)
PlotModel(Model)
SummarizeModel(Model)
PlotLoss(History)
PlotAcc(History)
