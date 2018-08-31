# Tiny-DSOD: Lightweight Object Detection for Resource Restricted Usage

This repository releases the code for our paper

**Tiny-DSOD: Lightweight Object Detection for Resource Restricted Usage (BMVC2018)**

**Yuxi Li, [Jianguo Li](https://sites.google.com/site/leeplus/), Jiuwei Li and Weiyao Lin**

The code is based on the [SSD](https://github.com/weiliu89/caffe/tree/ssd) and [DSOD](https://github.com/szq0214/DSOD) framework.

## Introduction

Tiny-DSOD tries to tackle the trade-off between detection accuracy and computation resource consumption. In this work, our tiny-model outperforms other small sized detection network (pelee, mobilenet-ssd or tiny-yolo) in the metrics of FLOPs, parameter size and accuracy. To be specific, on the dataset of PASCAL VOC2007, Tiny-DSOD achieves mAP of 72.1% with less than 1 million parameters (0.95M)

![basic unit DDB](https://github.com/lyxok1/Tiny-DSOD/raw/master/image/DDB.png)

![D-FPN](https://github.com/lyxok1/Tiny-DSOD/raw/master/image/front.png)

## Preparation

1. Install dependencies our caffe framework needs. You can visit the [caffe official website](http://caffe.berkeleyvision.org/installation.html) and follow the instructions there to install the dependent libraries and drivers correctly.

2. Clone this repository and compile the code
```Shell
git clone https://github.com/lyxok1/Tiny-DSOD.git

cd Tiny-DSOD
# visit the Makefile then modify the compile options and path to your library there
make -j8
```

3. Prepare corresponding dataset (if need training). Please see the document in [SSD](https://github.com/weiliu89/caffe/tree/ssd) detail 

## Train a model from scratch

Suppose the code is runing under the main directory of caffe.

First generate the model prototxt files
```Shell
python examples/DCOD/DCOD_pascal.py  # for voc training

python examples/DCOD/DCOD_kitti.py # for kitti training

python examples/DCOD/DCOD_coco.py # for coco training
```

And then use the binary `./build/tools/caffe` to train the generated network
```Shell
./jobs/DCOD300/${DATASET}/DCOD300_300x300/DCOD300_${DATASET}_DCOD300_300x300.sh
# Alternatively, you can directly use the binary to train in command line

./build/tools/caffe train -solver models/DCOD300/$DATASET/DCOD300_300x300/solver.prototxt -gpu all 2>&1 | tee models/DCOD300/$DATASET/DCOD300_300x300/train.log

```

## Deploy a pre-trained model

If you want to directly deploy a pre-trained model, you can use the demo scripts we provide in the `example/DCOD/` directory

- for image input detection, use the following command:
```Shell
python examples/DCOD/image_detection_demo.py <option>

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL          path to model prototxt file
  -weights WEIGHTS      path to weight file
  -img_dir IMG_DIR      path to input image
  -num NUM              number of images for detection
  -gpu                  specifiy using GPU or not
  -threshold THRESHOLD  threshold to filter bbox with low confidence
```

- for video input detection, use the following command:
```Shell
python examples/DCOD/video_detection_demo.py <option>

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL          path to model prototxt file
  -weights WEIGHTS      path to weight file
  -video VIDEO          path to input video
  -gpu                  specifiy using GPU or not
  -threshold THRESHOLD  threshold to filter bbox with low confidence
```

## Results

- Results on PASCAL VOC2007 (the models are trained on VOC07+12 trainval and test on VOC07 test)

| Method | # Params | FLOPs | *mAP* |
|:-------|:--------:|:-----:|:-----:|
| Faster-RCNN | 134.70M | 181.12B | 73.2 |
| SSD | 26.30M | 31.75B | 77.2 |
| Tiny-YOLO | 15.12M | 6.97B | 57.1 |
| MobileNet-SSD | 5.50M | 1.14B | 68.0|
| DSOD-smallest | 5.90M | 5.29B | 73.6|
| Pelee | 5.98M | 1.21B | 70.9 |
| Tiny-DSOD | 0.95M | 1.06B | 72.1 |

- Results on KITTI 2D Object Detction (the models are trained on half KITTI trainval and test on the other half)

| Method | # Params | FLOPs | car | cyclist | pedestrain |*mAP* |
|:-------|:--------:|:-----:|:---:|:-------:|:----------:|:----:|
| MS-CNN | 80M | - | 85.0 | 75.2 | 75.3 | 78.5 |
| FRCN | 121.2M | - | 86.0 | - | - | - |
| SqueezeDet | 1.98M | 9.7B | 82.9 | 76.8 | 70.4 | 76.7 |
| Tiny-DSOD | 0.85M | 4.1B | 88.3 | 73.6 | 69.1 | 77.0 |

- Results on COCO (the models are trained on trainval 135k and test on test-dev 2015)

| Method | # Params | FLOPs | *mAP(IOU 0.5:0.95)* |
|:-------|:--------:|:-----:|:-----:|
|MobileNet-v2+SSDLite| 4.30M | 0.80B | 22.1 |
|Pelee | 5.98M | 1.29B | 22.4 |
|Yolo-v2 | 67.43M | 34.36B | 21.6 |
|Tiny-DSOD | 1.15M | 1.12B | 23.2 |

## Released model

We released a model pretrained on VOC2007 on [Baidu Yun (3.8MB)](https://pan.baidu.com/s/1tNEZRWHwoVSOIuYwz2DlVQ)

## Example

![kitti1](https://github.com/lyxok1/Tiny-DSOD/raw/master/image/kitti1.png)

![kitti1](https://github.com/lyxok1/Tiny-DSOD/raw/master/image/kitti2.png)

![kitti1](https://github.com/lyxok1/Tiny-DSOD/raw/master/image/kitti3.png)

## Citation

If you think this work is helpful for your own research, please consider add following bibtex config in your latex file

```Latex
@inproceedings{li2018tiny,
  title = {{Tiny-DSOD}: Lightweight Object Detection for Resource-restricted Usage},
  author = {Yuxi Li, Jianguo Li, Jiuwei Li and Weiyao Lin},
  booktitle = {BMVC},
  year = {2018}
}

```
