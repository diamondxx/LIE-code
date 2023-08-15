# LIE-code
## Event-based Low-illumination Image Enhancement ([TMM](https://ieeexplore.ieee.org/abstract/document/10168206))

Welcome to the repository for our project on Event-based Low-illumination Image Enhancement. 

<img src="https://github.com/diamondxx/LIE-code/assets/37398726/62396571-8d63-4a79-9559-3e2b59922a15" width="60%">

## Pretrained Models
We provide the following pre-trained models for your use:

- Indoor model: [indoor_model](https://pan.baidu.com/s/1eEoMvo9WTNec1aFWROfZzw?pwd=tf5j)
- Outdoor model: [outdoor_model](https://pan.baidu.com/s/1ZlO3iGUu64cU2KovrEso2A?pwd=y7c0)
- Total model: [total_model](https://pan.baidu.com/s/1nkCfF-Sakl2__F3iqouh3A?pwd=cltf)

## Dataset
The LIE Dataset used in our research can be downloaded from [data](https://pan.baidu.com/s/1-rqECDLC8f_kONXV792-Iw?pwd=yi11).


<img src="https://github.com/diamondxx/LIE-code/assets/37398726/dacc9208-95f8-4d03-a42e-607ece01f7a7" width="60%">

## Test
To test the performance of our models, you can use the provided code.

Once you have configured the specific paths, you can run the following code. The pre-trained models should be placed in the `pretrained` directory, while the datasets and scenarios need to be stored in the `data` directory. 

The visualized results and metrics will be saved in the `result` directory.
```python
# Example code for testing
 python Test_Ours.py --resume ./pretrained/model_indoor.pth --data ./data/LIEDataset/orig_indoor_test --save ./result/display_indoor/
```

## Train
If you wish to train on your own data, please read and execute the `train.py` script, and modify the corresponding parameter settings. 

More detailed parameter configurations can be found in the `config` file.


## Acknowledgments
The codebase for this project is built upon the foundation of [pytorch-template](https://github.com/victoresque/pytorch-template) and [Restormer](https://github.com/swz30/Restormer)'s work.

## Citation

If you find our work useful, please consider citing our paper:
```
@ARTICLE{10168206,
  author={Jiang, Yu and Wang, Yuehang and Li, Siqi and Zhang, Yongji and Zhao, Minghao and Gao, Yue},
  journal={IEEE Transactions on Multimedia},
  title={Event-based Low-illumination Image Enhancement},
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMM.2023.3290432}
}
```
