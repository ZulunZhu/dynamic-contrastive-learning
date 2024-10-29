## Code for KDD 24 paper [Topology-monitorable Contrastive Learning on Dynamic Graphs](https://dl.acm.org/doi/10.1145/3637528.3671777).

## ![Framework](figures/overview.pdf) 

## This code is implemented based on  [InstantGNN](https://github.com/zheng-yp/InstantGNN.git) and [GGD](https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination.git).

## Requirements

- CUDA 10.1
- python 3.8.5
- pytorch 1.7.1
- GCC 5.4.0
- cython 0.29.21
- eigency 1.77
- numpy 1.18.1
- torch-geometric 1.6.3
- tqdm 4.56.0
- ogb 1.2.4
- [eigen 3.3.9] (https://gitlab.com/libeigen/eigen.git)

## Datasets

OGB Datasets can be downloaded from [here](https://ogb.stanford.edu). The website 'Open Graph Benchmark' provides an automatic method to download and convert the three datasets. So you can straightly run 'python convert_ogb.py' instead of downloading these datasets manually. We drop several edges to simulate the graphs' evolving nature. In the folder './convert/', we provide the codes to convert the three datasets.

## Compilation

Cython needs to be compiled before running, run this command:

```
python setup.py build_ext --inplace
```

## Running the code

- On OGB and Patent datasets

```
./all.sh
```

## Reference

If you find this repository useful in your research, please consider citing the following paper:

```
@inproceedings{zhu2024topology,
  title={Topology-monitorable Contrastive Learning on Dynamic Graphs},
  author={Zhu, Zulun and Wang, Kai and Liu, Haoyu and Li, Jintang and Luo, Siqiang},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={4700--4711},
  year={2024}
}
```
