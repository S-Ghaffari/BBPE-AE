import nltk.translate.bleu_score as bleu
import numpy as np

def GetBLEU(Passwords0:list[str], References0:list[str]) -> float:
    print(len(Passwords0))
    print(len(References0))
    
    Passwords = [[i for i in Password0] for Password0 in Passwords0]        #['m', 'i', 'l', 'u', 'p', 'e']
    #print((Passwords))                                                         #<class 'list'>
    References = [[i for i in References0] for References0 in References0]
    #print("*"*60)
    #print(type(References))
    
    mBLEUs = []
    for Password in Passwords:
        ts = []
        for Reference in References:
            t = bleu.sentence_bleu(references=[Reference],
                                   hypothesis=Password,
                                   weights=[0.25])
            ts.append(t)
        mBLEU = np.max(ts)
        mBLEUs.append(mBLEU)
    amBLEU = np.mean(mBLEUs)
    return amBLEU




with open(file='test_hotmail.txt', mode='r', encoding='UTF-8',errors='ignore') as F:
    Passwords = F.read().split('\n')
with open(file='pass-sample-bleu.txt', mode='r', encoding='UTF-8') as F:
    teTokens = F.read().split('\n')
amBLEU = GetBLEU(Passwords, teTokens)

print(f'Average Maximum BLEU: {amBLEU:.4f}')