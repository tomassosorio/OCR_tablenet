# TableNet

This repository consists on a Pytorch implementation of [TableNet](https://arxiv.org/abs/2001.01469).

To training or predict, you should first install the requirements by running the following code:

```bash
pip install -r requirements.txt
```

To train is only needed the `train.py` file which can be configured as wanted.
`marmot.py` and `tablenet.py` are inheritance of Pytorch Lighting modules: `LightningDataModule` and `LightningModule`, respectively.

To predict, it can be used the pre-trained weights already available and should be downloaded on the following link: [TableNet Weights](https://drive.google.com/drive/folders/1YbdQQ3ZLjrltfu7yBm7G5uVt2RYkWLoM?usp=sharing)

```bash
 python predict.py --model_weights='<weights path>' --image_path='<image path>'
```

or simply:
```bash
 python predict.py
```

To predict with the default image.
