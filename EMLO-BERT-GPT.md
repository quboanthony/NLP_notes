# BERT、GPT

## 基本概念

基本的出发点，如何让机器读懂人类的文字？我们需要将文字进行一定的编码，以便进行进一步的处理。在这方面，目前最先进的架构是BERT和GPT。这一篇笔记则旨在对两者进行一个总结。

BERT的全称为Bidirectional Encoder Representations from Transformers。BERT的结构，则是Transformer的encoder，其参数可以通过对大量没有标注文本进行训练来得到。关于transformer，之前的笔记则已经进行了总结，可以参阅transformer的笔记。

![alt BERT-0](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-0.png)

from [1]

总体来说，BERT做的事情，则是接受一个序列，输出这个序列每个节点的编码。同时这里李宏毅老师也提到一个trick，对于中文训练来说，带入BERT的序列以每一个字为节点，可能比一个词为节点更为恰当。因为相比单字，中文的词组合要更多，本身在输入到BERT之前，也要将每一个节点表示为One-Hot编码的形式再输入，词的组合的One-Hot向量就会非常长（除非有一定降维的准测）。

## 如何训练BERT？
由于BERT的架构已经再Transformer中进行了介绍，那么接下来，我们则比较关心这种embedding的结构，结合我们大量的无标注文本数据，都有哪些训练的方式呢？

- 第一种训练的方式

## 参考文献

[1] [李宏毅machine learning2020](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)
