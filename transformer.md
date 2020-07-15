# Transformer和Bert

## 基本概念
Transformer本质是一个seq2seq model，其中大量地使用了self_attention layer.

对于seq2seq中的处理序列的网络单元结构，最基本常见的，是RNN。

在序列的输入输出RNN的过程中，假设输入的序列是$a_0,a_1,a_2,\cdots,a_n$，对应输出的序列是$b_0,b_1,b_2,\cdots,b_n$。

如果RNN是单向的single directional，那么在输出$b_0$时，网络已经输入过$a_0$, 在输出$b_1$时，网络已经输入过$a_0,a_1$, 依次类推。

如果RNN时双向的bidirectional，那么即指在输出每一个$b_0,b_1,b_2,\cdots,b_n$的时候，每一次输出前，都已经输入过所有的$a_0,a_1,a_2,\cdots,a_n$。v

但RNN不容易并行化，且对于比较长的序列也会有早前的信息逐渐被弱化的情况。采用LSTM应该会有所缓解，但是attention结构的发明进一步改进了这个问题。后来发展出的self-attention，则是完全取代了RNN，可以直接作为

谷歌在attention is all you need中提出self-attention的概念，可以取代RNN原来可以做的事情。可以简单认为self-attention是一种新的层，跟RNN一样，输入一个序列，输出一个序列。同时，具有双向RNN的特点，即输出序列的每一个节点输出的时候，网络就已经见过所有输入的序列节点信息了。

所谓的Transformer，即
## 网络结构与数学表达
这个Self-attention到底是什么样的结构？

1. 设输入的序列为$\{x_1,x_2,x_3,\cdots,x_n\}$，那么每一层首先经过一轮embedding的映射，即矩阵相乘$a_i=Wx_i$，将$x_i$转化为$\{a_1,a_2,a_3,\cdots,a_n\}$，然后输入self-attention 层当中。


2. 在self-attention中，则将$a_i$乘以三种不同的transformation/matrix，产生三种不同的vector。这三种vector分别以$q_i,k_i,v_i$。这三个vector的含义（功能）分别为：
   
   - q: query（用来跟其他的序列节点进行匹配）
   $$
   q_i=W_qa_i
   $$

   - k: key (用来被q匹配的权重)
   $$
   k_i=W_k a_i
   $$

   - v: 代表该节点被抽取的信息权重
   $$
   v_i=W_v a_i
   $$

3. 所谓的匹配，是怎么匹配呢？接下来，
