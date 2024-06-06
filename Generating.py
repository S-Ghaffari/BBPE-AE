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
        with open(file=Path, mode='r', encoding='UTF-8') as F:
            S = F.read()
        Passwords = S.split('\n')#[:10000]
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

def LoadModel() -> mod.Sequential:
    Model = mod.load_model(filepath='Model', compile=True)
    return Model

def ResetSettings(RandomState:int,
                  Style:str) -> None:
    ran.seed(a=RandomState)
    plt.style.use(style=Style)
    np.random.seed(seed=RandomState)
    tf.random.set_seed(seed=RandomState)
    os.environ['PYTHONHASHSEED'] = str(RandomState)

def GenerateSingle(Model:mod.Sequential,
                   Vocabulary:dict,
                   Characters:dict,
                   nLag:int,
                   iVocabulary:dict) -> str:
    Tokens = ['<PADDING>' for _ in range(nLag)]
    StartIndex1 = Vocabulary['<START>']
    OOVIndex1 = Vocabulary['<OOV>']
    PaddingIndex1 = Vocabulary['<PADDING>']
    MaxLength2 = Model.input[1].shape[1]  
    First = True
    while True:
        InputTokens1 = Tokens[-nLag:]       
        InputIndexes1 = [Vocabulary[InputToken1] for InputToken1 in InputTokens1]       
        InputIndexes2 = []
        for Token1 in InputTokens1:
            if Token1 in ['<START>', '<END>', '<OOV>', '<PADDING>']:
                InputIndexes2.append(Characters[Token1])
            else:
                for Character in Token1:
                    InputIndexes2.append(Characters.get(Character, Characters['<OOV>']))
        while len(InputIndexes2) < MaxLength2:
            InputIndexes2 = [Characters['<PADDING>']] + InputIndexes2
        X = (np.array([InputIndexes1]), np.array([InputIndexes2]), np.array([InputIndexes1]), np.array([InputIndexes2]))
        Probabilities = Model.predict(X, batch_size=1, verbose=0)[0, -1]   
        Probabilities[[OOVIndex1, PaddingIndex1]] = 0                     
        if not First:
            Probabilities[StartIndex1] = 0
        else:
            First = False
        Probabilities /= Probabilities.sum()
        OutputIndex = np.random.choice(a=len(Vocabulary), p=Probabilities)
        OutputToken = iVocabulary[OutputIndex]
        Tokens.append(OutputToken)
        if OutputToken == '<END>':
            break
    Password = ''.join(Tokens[nLag + 1:-1])
    return Password

def GeneratePasswords(N:int,
                      Model:mod.Sequential,
                      Vocabulary:dict,
                      Characters:dict,
                      nLag:int) -> None:
    iVocabulary = {v: k for k, v in Vocabulary.items()}
    Passwords = []
    for i in range(N):
        print(f'Generating Password {i + 1} / {N}')
        Password = GenerateSingle(Model, Vocabulary, Characters, nLag, iVocabulary)
        Passwords.append(Password)
    with open('Generated Passwords-10epoch-hotmail-1000.txt', mode='w', encoding='UTF-8') as F:
        F.write('\n'.join(Passwords))

RandomState = 0
Style = 'ggplot'
Path = 'train_hotmail.txt'
sVocabulary0 = 30000
nLag = 5
nHorizon = 1

ResetSettings(RandomState, Style)
_, _, Vocabulary, Characters = Fetch(Path, sVocabulary0)
Model = LoadModel()
GeneratePasswords(1000, Model, Vocabulary, Characters, nLag)
