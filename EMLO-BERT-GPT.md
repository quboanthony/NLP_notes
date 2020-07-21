# BERT、GPT

## BERT基本概念

基本的出发点，如何让机器读懂人类的文字？我们需要将文字进行一定的编码，以便进行进一步的处理。在这方面，目前最先进的架构是BERT和GPT。这一篇笔记则旨在对两者进行一个总结。

BERT的全称为Bidirectional Encoder Representations from Transformers。BERT的结构，则是Transformer的encoder，其参数可以通过对大量没有标注文本进行训练来得到。关于transformer，之前的笔记则已经进行了总结，可以参阅transformer的笔记。

![alt BERT-0](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-0.png)

from [1]

总体来说，BERT做的事情，则是接受一个序列，输出这个序列每个节点的编码。同时这里李宏毅老师也提到一个trick，对于中文训练来说，带入BERT的序列以每一个字为节点，可能比一个词为节点更为恰当。因为相比单字，中文的词组合要更多，本身在输入到BERT之前，也要将每一个节点表示为One-Hot编码的形式再输入，词的组合的One-Hot向量就会非常长（除非有一定降维的准测）。

## 如何训练BERT？
由于BERT的架构已经再Transformer中进行了介绍，那么我们则比较关心这种embedding的结构，都有哪些训练的方式？


- 无监督训练
  1. Masked LM "完形填空": 一个序列约15%的词汇会被盖住，训练bert把盖住的词汇填回来。我们输入一个序列，对于盖住的词对应的输出加入一个线性的多分类器，来计算预测值和损失函数的反馈。由于线性多分类器非常弱，所以这就要求bert必须抽出比较好的编码表达，从而让线性多分类器进行预测。
  - 如果两个词填在同一个地方没有违和感，则两个词具有相似的embedding
  - ![alt BERT-1](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-1.png)

  2. Next Sentence "预测是否是接着的下一句"：此时对于输入序列，需要增加两个节点，[CLS]表示判别节点，用来输出被分割的两句是否构成上下文的关系，[SEP]表示分割节点，用来指示分割上下文的节点。
  - [CLS]放在句子的开口还是结尾还是中间，对于基于transformer的bert，关系不大。
  - ![alt BERT-2](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-2.png)

- 一般两种方式同时使用，用来训练bert的embedding结构。

## 如何使用BERT？
1. 直接使用bert一起学习文本分类任务，结合线性分类器，此时bert最好已经有一定预训练过的参数，在分类任务中再fine-tune。
- ![alt BERT-3](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-3.png)
2. 带入bert进行slot filling训练，即输入一个句子，输出句子中的每一个词，属于什么类别。同样可以结合线性分类器。
- ![alt BERT-4](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-4.png)
3. 带入bert学习Natural Language Inference自然语言推论。输入一个premise前提，让机器学习这个前提下的假设是否成立或是未知。
- ![alt BERT-5](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-5.png)
4. 带入bert进行Extraction-based QA抽取式的问答系统的训练。想要达到的效果是，输入一篇文章，然后输入想问的问题，希望机器根据文章给出答案。不过Extraction-based意思是我们假设问题的答案一定出现在文章里。
- 输入有文章和对应的问题，输出有2个整数，两个整数表示答案在文章的第几个词到第几个词。

- ![alt BERT-6](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-6.png)
- 输入的形式是如何呢？同样引入[CLS]和[SEP]，分割问题的序列和文章的序列，输出embedding。
- 然后对于输出，再学习两个跟输出同大小的向量。比如ppt中红色的vector，跟每一个embedding做dot product，再输出softmax来预测答案的第一个词的位置。然后蓝色的vector同样跟每一个embedding做dot product，输出softmax来预测答案的最后一个词位。
- 那么当开头的位置跟结尾的位置预测结果是矛盾的时候，则代表没有答案。
- ![alt BERT-7](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-7.png)
- ![alt BERT-8](EMLO-BERT-GPT-figs/../EMLO-BERT-GPT-figs/Bert-8.png)
## 参考文献

[1] [李宏毅machine learning2020](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)

[2]
