# Official implementation codes for the paper "PathGAN: Local Path Planning with Generative Adversarial Networks"

The paper was submitted to CVPR 2021 and is now under review.

An old version of the paper (which may be quite different from the submitted one) can be found at https://arxiv.org/abs/2007.03877.  
ETRIDriving, the autonomous driving dataset for the training and the evaluation, can be found at https://github.com/d1024choi/etridriving.

## Preparation
1) DATASET  
    * Create a folder (for example, /home/dooseop/DATASET/) and copy the driving sequences (link) listed in './dataset/train.txt' and './dataset/test.txt' into the created folder.    
    * Copy 'preprocessed_dataset.cpkl' (link) into './dataset/'.  

2) Pretrained Network Params  
    * Create a folder './pretrained_cnn' and copy 'saved_cnn_exp12_model70.pt' (link) into the created folder.  

3) Libraries
    * python 3.6  
    * pytorch 1.1.0  
    * torchvision 0.3.0  
    * opencv 3.1 >=  
    * scipy 1.3.0 >=  
    * scikit-image 0.14 >=
    * scikit-learn 0.22 >=  
  
## Train New Models
To train the model from scratch, run the followings. The parameters of the trained networks will be stored at the folder './saved_models/model0'.
```sh
$ python train.py --exp_id 0
```
