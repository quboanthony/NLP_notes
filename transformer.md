# Transformer和Bert

Transformer本质是一个seq2seq model，其中大量地使用了self_attention layer.

对于seq2seq中的处理序列的网络单元结构，最基本常见的，是RNN。

在序列的输入输出RNN的过程中，假设输入的序列是$a_0,a_1,a_2,\cdots,a_n$，对应输出的序列是$b_0,b_1,b_2,\cdots,b_n$。

如果RNN是单向的single directional，那么在输出$b_0$时，网络已经输入过$a_0$, 在输出$b_1$时，网络已经输入过$a_0,a_1$, 依次类推。

如果RNN时双向的bidirectional，那么即指在输出每一个$b_0,b_1,b_2,\cdots,b_n$的时候，每一次输出前，都已经输入过所有的$a_0,a_1,a_2,\cdots,a_n$。

