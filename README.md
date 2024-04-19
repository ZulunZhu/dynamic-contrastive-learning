# Topology-aware Graph Contrastive Learning on Dynamic Graphs

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

