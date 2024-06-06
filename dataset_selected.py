import random

with open('hotmail.txt', 'r',encoding='UTF-8', errors='ignore') as file:
    lines = file.readlines()
b = random.sample(lines, k=11111)
test_size = int(len(b) * 0.2)
random.shuffle(b) 
test = b[:test_size]
train = b[test_size:]

with open('test_hotmail.txt', 'w', errors='ignore') as file:
    file.writelines(test)

with open('train_hotmail.txt', 'w', errors='ignore') as file:
    file.writelines(train)
