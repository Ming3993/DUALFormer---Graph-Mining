# DUALFORMER: DUAL GRAPH TRANSFORMER 

DUALFormer is a lightweight Graph Transformer (GT) architecture that utilizes a dual-dimensional design to enable efficient and effective learning of graph representations. It addresses two key issues in existing GTs: the scalability issue on large-scale graphs and the tradeoff dilemma between local and global information. 

## Architecture
![Motivation and Design](design_motivation.png)
<b>Figure:</b> The design motivation of the proposed DUALFormer and its comparison with existing state-of-the-art GT architectures. (a) Existing GTs suffer from two primary challenges: 1) the scalability issue from Self-Attention (SA) mechanisms, and 2) the tradeoff dilemma of local and global information. By default, global SA mechanisms serve the value V as the agent representations of node features, so that by employing attention score matrix sim(Q, K) ∈ Rn×n to capture global dependencies among nodes. Intuitively, by leveraging the approximation sim(Q, K)V ≈ Qsim(K, V), and treating the query Q as the agent representations, the above global SA mechanism can be efficiently implemented in the feature dimension. (b) DUALFormer is a dual-dimensional GT architecture that seamlessly integrates the local GNN block and global SA block on dual dimensions. Thus, DUALFormer effectively and comprehensively leverages the advantages of both dimensions.

Extensive experiments across graph benchmark datasets demonstrate the superiority of DUALFormer compared to existing Graph Neural Networks (GNNs) and GTs. For more details, please refer to our [[Paper]](https://openreview.net/pdf?id=4v4RcAODj9)





# Cite

If you find DUALFormer useful in your research or work, please cite our paper. 
```
> @inproceedings{DUALFormer,
  title={DUALFormer: Dual Graph Transformer},
  author={Zhuo, Jiaming and Liu, Yuwei and Lu, Yintong and Ma, Ziyi and Fu, Kun and Wang, Chuan and Guo, Yuanfang and Wang, Zhen and Cao, Xiaochun and Yang, Liang},
  booktitle={{ICLR}},
  year={2025},
}
```
