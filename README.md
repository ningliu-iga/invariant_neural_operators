# Invariant Neural Operators
This repository houses the code for the following paper:
- [INO: Invariant Neural Operators for Learning Complex Physical Systems with Momentum Conservation](https://arxiv.org/abs/2212.14365)

**Abstract**: Neural operators, which emerge as implicit solution operators of hidden governing equations, have recently become popular tools for learning responses of complex real-world physical systems. Nevertheless, the majority of neural operator applications has thus far been data-driven, which neglects the intrinsic preservation of fundamental physical laws in data. In this paper, we introduce a novel integral neural operator architecture, to learn physical models with fundamental conservation laws automatically guaranteed. In particular, by replacing the frame-dependent position information with its invariant counterpart in the kernel space, the proposed neural operator is by design translation- and rotation-invariant, and consequently abides by the conservation laws of linear and angular momentums. As applications, we demonstrate the expressivity and efficacy of our model in learning complex material behaviors from both synthetic and experimental datasets, and show that, by automatically satisfying these essential physical laws, our learned neural operator is not only generalizable in handling translated and rotated datasets, but also achieves state-of-the-art accuracy and efficiency as compared to baseline neural operator models. %Therefore, our INO can serve as a more efficient and robust model for learning real-world mechanical systems.

## Requirements
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)


## Running experiments
To run the 2nd example in the INO paper
```
python3 ino_lps_circular.py
```

## Datasets
We provide the linear peridynamic solid (LPS) and tissue datasets that are used in the paper. The data generation can be found in the appendix of the paper. The data are given in the form of matlab files, which can be loaded using the scripts provided in utils.py. 

- LPS dataset: see the data_linear_peridynamic_solid subfolder
- [Tissue dataset](https://drive.google.com/drive/folders/1BQjPEDYRJv5VjZ_bTyK9OScPKVkcdt3y)

## Citation

```
@inproceedings{liu2022ino,
  title={INO: Invariant Neural Operators for Learning Complex Physical Systems with Momentum Conservation},
  author={Liu, Ning and Yu, Yue and You, Huaiqian and Tatikola, Neeraj},
  booktitle={Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS) 2023}
}
```
