# Char-RNN
一个[char-rnn](https://github.com/karpathy/char-rnn)项目的Keras实现。  
修改自[Keras官方示例](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)。  
使用了GRU而不是LSTM作为神经网络的隐藏层，大大提高了训练速度，并为代码添加了详细的中英注释。  
在测试输出时，提供了Sample和Max两种取值方式。
# Todo
* 文本分批次训练(减少内存占用)