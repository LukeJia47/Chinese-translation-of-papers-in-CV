# Transformer架构

![image-20220916090806763](注意力机制与Transformer.assets/image-20220916090806763.png)

Transformer 模型的架构就是一个 seq2seq 架构，由多个 Encoder Decoder 堆叠而成。

**Encoder:**编码器由 N = 6 个相同的层组成。每层有两个子层。第一个是多头自注意力机制，第二个是简单的、按位置的全连接前馈网络。我们在两个子层中的每一个周围使用残差连接 [11]，然后进行层归一化 [1]。即每个子层的输出为LayerNorm(x + Sublayer(x))，其中Sublayer(x)是子层自己实现的函数。为了促进这些残差连接，模型中的所有子层以及嵌入层都会产生维度 ![image-20220906145443659](注意力机制与Transformer.assets/image-20220906145443659.png)= 512 的输出。

**Decoder:**解码器也由一堆 N = 6 个相同的层组成。除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力。与编码器类似，我们在每个子层周围使用残差连接，然后进行层归一化。我们还修改了解码器堆栈中的自注意力子层，以防止位置关注后续位置。这种mask与输出嵌入偏移一个位置的事实相结合，确保对位置 i 的预测只能依赖于位置小于 i 的已知输出。

概括来说，我们输入法语：je suis étudiant，经过六个 Encoder 后得到了类似于 Context Vector 的东西，然后将得到的向量放进 Decoder 中，每个 Decoder 会对上一个 Decoder 的输出进行 Self-Attention 处理，同时会把得到的结果与 Encoder 传递过来的 Vector 进行 Encoder-Decoder Attention 处理，将结果放入前馈网络中，这算是一个 Decoder，而把六个 Decoder 叠加起来学习，便可得到最后的结果。这里叠起来的编解码器的数量不是固定的，至于 Encoder 和 Decoder 的工作原理在下面章节介绍。

# Self-Attention(自注意力)

![image-20220916093845350](注意力机制与Transformer.assets/image-20220916093845350.png)

## 缩放点积注意力

![image-20220916094012060](注意力机制与Transformer.assets/image-20220916094012060.png)

## 多头注意力

![image-20220916094030244](注意力机制与Transformer.assets/image-20220916094030244.png)

其中投影是参数矩阵![image-20220907092624453](注意力机制与Transformer.assets/image-20220907092624453.png)![image-20220907092647799](注意力机制与Transformer.assets/image-20220907092647799.png)

在这项工作中，我们使用 h = 8 个并行注意力层或头。对于其中的每一个，我们使用![image-20220907092856519](注意力机制与Transformer.assets/image-20220907092856519.png)。由于每个头的维度减少，总计算成本类似于具有全维度的单头注意力。

![image-20220916094511283](注意力机制与Transformer.assets/image-20220916094511283.png)

## 理解缩放点积注意力

![image-20220916094637316](注意力机制与Transformer.assets/image-20220916094637316.png)

![image-20220916094827873](注意力机制与Transformer.assets/image-20220916094827873.png)

![image-20220916095403177](注意力机制与Transformer.assets/image-20220916095403177.png)

![image-20220916095427063](注意力机制与Transformer.assets/image-20220916095427063.png)

![image-20220916095525036](注意力机制与Transformer.assets/image-20220916095525036.png)

### 向量化

![image-20220916095642075](注意力机制与Transformer.assets/image-20220916095642075.png)

![image-20220916095740316](注意力机制与Transformer.assets/image-20220916095740316.png)

![image-20220916095905334](注意力机制与Transformer.assets/image-20220916095905334.png)

## 理解多头注意力

![image-20220916100024452](注意力机制与Transformer.assets/image-20220916100024452.png)

![image-20220916100244297](注意力机制与Transformer.assets/image-20220916100244297.png)

![image-20220916100327236](注意力机制与Transformer.assets/image-20220916100327236.png)

![image-20220916100434226](注意力机制与Transformer.assets/image-20220916100434226.png)

下图是多头注意力全部流程

![image-20220916100535237](注意力机制与Transformer.assets/image-20220916100535237.png)

## 理解前馈网络和层归一化

### Feed Forward

![image-20220916101018406](注意力机制与Transformer.assets/image-20220916101018406.png)

### Layer normalization

![image-20220916101353502](注意力机制与Transformer.assets/image-20220916101353502.png)

![image-20220916101503972](注意力机制与Transformer.assets/image-20220916101503972.png)

## 总结

![image-20220916101644921](注意力机制与Transformer.assets/image-20220916101644921.png)

## Positional Encoding(位置编码)

在这项工作中，我们使用不同频率的正弦和余弦函数：

![image-20220907164435998](注意力机制与Transformer.assets/image-20220907164435998.png)

其中pos是位置，i是维度。也就是说，位置编码的每个维度对应一个正弦曲线。波长形成从 2π 到 10000 · 2π 的几何级数。我们选择这个函数是因为我们假设它可以让模型轻松学习通过相对位置来参与，因为对于任何固定的偏移量 k，![image-20220907164755699](注意力机制与Transformer.assets/image-20220907164755699.png)可以表示为![image-20220907164819418](注意力机制与Transformer.assets/image-20220907164819418.png)的线性函数。

我们还尝试使用学习的位置embeddings [9]，发现这两个版本产生了几乎相同的结果（见表 3 行 (E)）。我们选择了正弦版本，因为它可以让模型推断出比训练期间遇到的序列长度更长的序列长度。

## 理解位置编码

![image-20220916105231022](注意力机制与Transformer.assets/image-20220916105231022.png)

![image-20220916105353850](注意力机制与Transformer.assets/image-20220916105353850.png)

![image-20220916105521113](注意力机制与Transformer.assets/image-20220916105521113.png)

![image-20220916105746618](注意力机制与Transformer.assets/image-20220916105746618.png)

偶数维度(0,2,4,...)使用如下公式:

![image-20220916105830499](注意力机制与Transformer.assets/image-20220916105830499.png)

奇数维度采样如下公式:

![image-20220916105901995](注意力机制与Transformer.assets/image-20220916105901995.png)

![image-20220916110021163](注意力机制与Transformer.assets/image-20220916110021163.png)

![image-20220916110113521](注意力机制与Transformer.assets/image-20220916110113521.png)

![image-20220916110218809](注意力机制与Transformer.assets/image-20220916110218809.png)

![image-20220916110241346](注意力机制与Transformer.assets/image-20220916110241346.png)

# Decoder(解码器)

![image-20220916150751960](注意力机制与Transformer.assets/image-20220916150751960.png)

![image-20220916172542925](注意力机制与Transformer.assets/image-20220916172542925.png)

![image-20220916150816875](注意力机制与Transformer.assets/image-20220916150816875.png)

![image-20220916151025320](注意力机制与Transformer.assets/image-20220916151025320.png)

![image-20220916151109680](注意力机制与Transformer.assets/image-20220916151109680.png)

# Mask

![image-20220916151203033](注意力机制与Transformer.assets/image-20220916151203033.png)

![image-20220916151219048](注意力机制与Transformer.assets/image-20220916151219048.png)

## Padding Mask

![image-20220916151344164](注意力机制与Transformer.assets/image-20220916151344164.png)

## Sequence Mask

![image-20220916151645864](注意力机制与Transformer.assets/image-20220916151645864.png)

![image-20220916174322774](注意力机制与Transformer.assets/image-20220916174322774.png)

# 参考链接

https://zhuanlan.zhihu.com/p/264468193