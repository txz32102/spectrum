# how to run the code?

activate the conda env first

```bash
source /home/data1/anaconda3/bin/activate
conda activate musong_base
```

then you can located at the `/home/musong/workspace/python/spectrum/2024_10_31_16_25_resnet34.ipynb` for training resnet, 

and run `/home/musong/workspace/python/spectrum/2024_10_31_16_35_resnet_attention.ipynb` for training resnet+attention.

the data is located at `/home/musong/workspace/python/spectrum/data`, its structure is like

```bash
(musong_base) ➜  data git:(main) ✗ tree
.
├── X.npy
├── X_toy.npy
├── Y.npy
├── Y_toy.npy
├── Z.npy
└── Z_toy.npy

0 directories, 6 files
```

where `X.npy` is the training data of shape `(batch_size, spectrum_len, value)`, `value` contains the real part and the image part, it is a degree range from 0 to 360, but there may exists some values below 0 and exceed 360. 

`Y.npy` is the expected value of the spectrum real part

`Z.npy` is the label, which is correspond to the `value` which is in two values , namely `(ph0, ph1)`, these two are float degrees


## zj train

/home/data1/zj/envs/phaseC/bin/python3.7 /home/data1/zj/PhaseCorrection/pythonCode/attention_resnet_train.py

## debug01 train

cd /home/data1/zj/PhaseCorrection/pythonCode && /home/data1/zj/envs/phaseC/bin/python3.7 /home/data1/zj/PhaseCorrection/pythonCode/chk_test/debug01.py

## attention resnet train

/home/data1/zj/envs/phaseC/bin/python3.7 /home/data1/zj/PhaseCorrection/pythonCode/attention_resnet_train.py

## grant permission 

# conda env

conda activate /home/data1/zj/envs/phaseC


# table

| Statistic              | X                     | Z                     |
|------------------------|-----------------------|-----------------------|
| Mean                   | 0.2998                | 92.7306               |
| Standard Deviation     | 31.1594               | 119.2016              |
| Min                    | -2455.8848            | -46.0863              |
| Max                    | 2456.2163             | 375.1798              |
| Shape                  | (18480, 32768, 2)     | (18480, 1, 2)         |
