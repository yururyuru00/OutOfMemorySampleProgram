# OutOfMemorySampleProgram
RAM memory keeps increasing without being freed every time python script repeated by grid-search...

## Requirements
- torch
- torch-cluster    
- torch-scatter    
- torch-sparse     
- torch-spline-conv
- torch-geometric
- ogb
- hydra

```bash
pip install -r requirements.txt
```

## How to Reproduct Out Of Memory
```bash
python3 train.py -m 'NN.learning_rate=choice(0.01,0.001)' 'NN.n_hid=choice(64,128,256)' 'NN.n_layer=range(3,6)'
```