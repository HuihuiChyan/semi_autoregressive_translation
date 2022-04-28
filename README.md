基于fairseq实现的半自回归模型，主要参考了论文《Semi-Autoregressive Neural Machine Translation》；

每次解码N个token，这样速度有很大提升，但是翻译性能有下降；

改动sequence generator花费了很长时间，然后跑通了之后就没再管了，所以可能有些小BUG？
