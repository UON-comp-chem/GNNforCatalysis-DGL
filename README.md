# Graph Neural Networks for Catalysis
Graph Nerual Network for Catalysis based on Deep Graph Library

## Installation

**Requirements**

**1. Python environment**

We recommend using Conda package manager

```bash
conda create -n catgnn python=3.7.3
source activate catgnn
```

**2. Essential python packages**
  - dgl=0.5.3
  - pytorch=1.6.0
  - pymatgen=2020.10.20
  - scikit-learn=0.23.2
  - ase=3.20.1

**3. Installing GNNforCatalysis-DGL**
Add this manually to the system path
```bash
import sys
sys.path.append(('$PWD'))
```


## Acknowledgements
- Contain codes from [Alchemy](https://github.com/tencent-alchemy/Alchemy), [DGL-LifeSci](https://github.com/awslabs/dgl-lifesci), 
  [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack), [GASpy](https://github.com/ulissigroup/GASpy)
