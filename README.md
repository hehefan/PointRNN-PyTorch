# [PointRNN: Point Recurrent Neural Network for Moving Point Cloud Processing](https://arxiv.org/pdf/1910.08287.pdf)

## Citation
If you find our work useful in your research, please consider citing:
```
@article{fan19pointrnn,
  author    = {Hehe Fan and Yi Yang},
  title     = {PointRNN: Point Recurrent Neural Network for Moving Point Cloud Processing},
  journal   = {arXiv},
  volume    = {1910.08287},
  year      = {2019}
}
```
## License
The code is released under MIT License.
## Installation
Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413):
```
    cd modules
    python setup.py install
```
To see if the compilation is successful, try to run `python modules/pointrnn_cell_impl.py` to see if a forward pass works.
## Related Repos
1. PointRNN TensorFlow implementation: https://github.com/hehefan/PointRNN
2. PointNet++ PyTorch implementation: https://github.com/erikwijmans/Pointnet2_PyTorch
