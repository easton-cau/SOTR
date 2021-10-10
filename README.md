# SOTR: Segmenting Objects with Transformers [ICCV 2021]
By [Ruohao Guo](https://github.com/easton-cau), Dantong Niu, [Liao Qu](https://github.com/QuLiao1117), Zhenbo Li

## Introduction

This is the official implementation of  [SOTR](https://arxiv.org/abs/2108.06747).

<img src="images/overview.png" alt="image" style="zoom:60%;" />



## Models

### COCO Instance Segmentation Baselines with SOTR

Name |  mask AP | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | download
:-- |:---:|:---:|:---:|:---:|:---:
[SOTR_R50](configs/SOTR/R50.yaml) | 39.6 | 10.3 | 58.7 | 72.1 | [model](https://drive.google.com/file/d/1Aijh9VYy0Omjsz2ei-DfuncaBO0qbct7/view?usp=sharing)
[SOTR_R101](configs/SOTR/R101.yaml) | 40.2 | 10.2 | 59.0 | 73.1 | [model](https://drive.google.com/file/d/1CzQTsvn9vxLnFkDJpIlitFXu1X_vw1dZ/view?usp=sharing)
[SOTR_R101_DCN](configs/SOTR/R_101_DCN.yaml) | 42.0 | 11.4 | 60.7 | 74.5| [model](https://drive.google.com/file/d/19Dy6sXrwaNwGwNvuQyv5pZMWGM_at0ym/view?usp=sharing) 

## Installation & Quick start

- First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).


- Then build SOTR with:


```
https://github.com/easton-cau/SOTR
cd SOTR
python setup.py build develop
```

- Then follow [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md)  to set up the datasets (e.g., MS-COCO).

- Evaluating

  - Download the trained models for COCO. 

  - Run the following command

    ```
    python tools/train_net.py \
        --config-file configs/SOTR/R101.yaml \
        --eval-only \
        --num-gpus 4 \
        MODEL.WEIGHTS work_dir/SOTR_R101/SOTR_R101.pth
    ```

- Training

  - Run the following command

    ```
    python tools/train_net.py \
        --config-file configs/SOTR/R101.yaml \
        --num-gpus 4 \
    ```


## Acknowledgement

Thanks [Detectron2](https://github.com/facebookresearch/detectron2) and [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) contribution to the community!

The work is supported by National Key R&D Program of China (2020YFD0900204) and Key-Area Research and Development Program of Guangdong  Province China (2020B0202010009).

## FAQ
If you want to improve the usability or any piece of advice, please feel free to contant directly (ruohguo@foxmail.com).

## Citation

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@inproceedings{guo2021sotr,
  title={SOTR: Segmenting Objects with Transformers},
  author={Guo, Ruohao and Niu, Dantong and Qu, Liao and Li, Zhenbo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7157--7166},
  year={2021}
}
```

