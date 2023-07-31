# WCST-ML, by Luca Scimeca

The code in this repository has been developed as part of the paper https://openreview.net/pdf?id=qRDQi3ocgR3 . Withing ths "src" folder you can find several examples on how to test multiple deep learning architectures on fully correlated datasets to test model bias. This is not an official release. 

* To know more about the project or collaborate please contact me at luca_scimeca@dfci.harvard.edu

## Installation

The code was tested on pycharm in an environment with the following dependencies:

* python 3.9.1
* pytorch 1.8.0
* tensorboard 2.4.1
* timm 0.4.5


## Usage

The src folder contains all code for the project. in 'src/utils/convert/' you can find useful python scripts to transform DSprites and UTKFace datasets into fully correlated versions readily useful for the WCST-ML test. The Mode connectivity and loss-landscape invertigations are based on the respective paper repositories. 



## License
[GPL](https://www.gnu.org/licenses/#GPL)


To cite the paper please use:
```
@inproceedings{scimeca2022shortcut,
  title = {Which Shortcut Cues Will DNNs Choose? A Study from the Parameter-Space Perspective},
  author = {Scimeca, Luca and Oh, Seong Joon and Chun, Sanghyuk and Poli, Michael and Yun, Sangdoo},
  booktitle = {International Conference on Learning Representations},
  year = {2022}
}
```


