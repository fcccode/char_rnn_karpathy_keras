# coding: utf-8
from __future__ import print_function
from keras.layers.recurrent import GRU
from tensorflow.python.lib.io.file_io import file_exists
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
import sys
from time import sleep
import psutil
from keras.layers.wrappers import TimeDistributed
####
# minesh.mathew@gmail.com
# modified version of text generation example in keras; trained in a many-to-many fashion using a time distributed dense layer
####
'''
w568w 1278297578@qq.com
添加中文注释，修改为Keras 2 API
Add code comments in Chinese,change the code to Keras 2 API and replace LSTM with GRU.
'''



def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
# 测试
# 生成结果，使用max法，易出现重复，考虑使用sample
# testing
# now you use the trained model to generate text.
def test(seed_string, length):
    print ('The generated text is')
    sys.stdout.write(seed_string)
    for i in range(length):
        x = np.zeros((1, len(seed_string), len(chars)))
        for t, char in enumerate(seed_string):
            x[0, t, char_indices[char]] = 1.
        preds = model.predict(x, verbose=0)[0]
        #next_index = np.argmax(preds[len(seed_string) - 1])
        next_index = sample(preds[len(seed_string) - 1])
        next_char = indices_char[next_index]
        seed_string = seed_string + next_char
    
        sys.stdout.write(next_char)


# 打开文件
# Read file
text = open('./textdatasets/tiny_input.txt').read().lower()
print('text length:', len(text))
# 处理中文编码问题
# deal with encoding problem when processing non-ascii chars.
text = unicode(text, 'utf-8')
# 取出字符集合(重复字符只包含一次)，作为词字典
# put every word into a set,which's used as a word dictionary
chars = sorted(list(set(text)))
print('total chars:', len(chars))
# 生成字典，获得字符在chars中的位置
# Eg. char_indices['c'] will return the index of 'c' in chars array
char_indices = dict((c, i) for i, c in enumerate(chars))
# 生成字典，获得下标在chars中的字符
# Eg. indices_char[0] will return the first element in chars array
indices_char = dict((i, c) for i, c in enumerate(chars))

# 40个40个字符地分割文本为sentences，next_chars为sentences向后移动一位文本
# 如：文本为ABCDE,maxlen=3则
# sentences=[[A,B,C],[B,C,D]]
# nextchars=[[B,C,D],[C,D,E]]

# split the text into sequences of length=maxlen
# input is a sequence of 20 chars and target is also a sequence of 40 chars shifted by one position
# Eg. if you maxlen=3 and the text is abcdefghi, your input ---> target pairs will be
# [a,b,c] --> [b,c,d], [b,c,d]--->[c,d,e]....and so on
maxlen = 20
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen + 1, 1):
    sentences.append(text[i: i + maxlen]) 
    next_chars.append(text[i + 1:i + 1 + maxlen]) 
print('sequences length:', len(sentences))
# 向量化
# Vectorize the data
print()
print('Vectorization...')
#输出两个巨型矩阵的大小及可用内存
#print the size of these two huge martixs
print('array size:', float((len(sentences) * maxlen * len(chars))) * 2 / 1024 / 1024, 'MB')
info = psutil.virtual_memory()
print('free memory:', float(info.available / (1024 ** 2)), 'MB')
# 输入格式为三维向量，分别代表(输入组量,step长度,每个step的输入长度)
# 这里使用one hot向量，x[0,0,0]=1，表示第1句话的第1个字符是chars[0]
# Input is a 3-dim vector,shape means (data_size,step_length,input_length_at_each_step)
# X and Y are one-hot vectors,
# eg. x[0,0,0]=1 means the first char of the first sentence is char[0]
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)  
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1

for i, sentence in enumerate(next_chars):
    for t, char in enumerate(sentence):
        y[i, t, char_indices[char]] = 1

print ('vetorization completed')
# 建构模型：GRU+20%丢弃率+时间步的Full-connect网络
# build the model: 1 stacked GRU+20% DropOut+Time Distributed Full-Connected Dense
print('Build model...')
model = Sequential()
# 指定了input长度
# Specify the input length
model.add(GRU(512, input_shape=(None, len(chars)), return_sequences=True)) 
# model.add(LSTM(512, return_sequences=True)) 
model.add(Dropout(0.2))
# TimeDistributed就是在每个时间步上均执行一次Dense，即many to many，这里起修饰LSTM层输出结果作用
# Dense(len(chars))代表输出的第三个维度为chars长度
model.add(TimeDistributed(Dense(len(chars))))
# 定义激活层
model.add(Activation('softmax'))
# rmsprop训练RNN很合适
# rmsprop is faster than other optimizers when training a RNN
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print (model.summary())
# 载入保存的参数
# load saved weights(if exist)
if file_exists('Karpathy_LSTM_weights.h5'):
    model.load_weights('Karpathy_LSTM_weights.h5')
# 训练
# Training
for iteration in range(1, 5):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # 拟合模型，128个batch为一组
    # train the model,batch size=128,epochs=1
    history = model.fit(X, y, batch_size=128, epochs=1)
    # 必须在IPython上延时0.1s，否则会在save_weights时出现I/O error
    # Must sleep on IPython,otherwise I/O error will be thrown when calling save_weights()
    sleep(0.1)  # https://github.com/fchollet/keras/issues/2110
    model.save_weights('Karpathy_LSTM_weights.h5', overwrite=True)
test(u"我",1000)
