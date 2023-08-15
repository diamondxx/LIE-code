# LIE-code
## Event-based Low-illumination Image Enhancement ([TMM](https://ieeexplore.ieee.org/abstract/document/10168206))

Welcome to the repository for our project on Event-based Low-illumination Image Enhancement. 

<img src="https://github.com/diamondxx/LIE-code/assets/37398726/62396571-8d63-4a79-9559-3e2b59922a15" width="70%x">

## Pretrained Models
We provide the following pre-trained models for your use:

- Indoor model: [indoor_model](link_to_indoor_model)
- Outdoor model: [outdoor_model](link_to_outdoor_model)
- Total model: [total_model](link_to_total_model)

## Dataset
The LIE Dataset used in our research can be downloaded from [data](link_to_dataset).


<img src="https://github.com/diamondxx/LIE-code/assets/37398726/dacc9208-95f8-4d03-a42e-607ece01f7a7" width="70%">

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

[Figure5.pdf](https://github.com/diamondxx/LIE-code/files/12342722/Figure5.pdf)

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
