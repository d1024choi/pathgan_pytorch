# The official implementation of the paper "PathGAN: Local Path Planning with Attentive Generative Adversarial Networks" published in ETRI Journal.

![fig1](./images/fig1.png)

Our path generation model is designed to generate multiple plausible paths that are consistent with the input driving intention and speed, from egocentric images.

The arxiv version of the paper can be found at https://arxiv.org/abs/2007.03877. ETRIDriving, the autonomous driving dataset for the training and the evaluation, can be found at https://github.com/d1024choi/ETRIdriving-DevKit.

## Preparation
1) DATASET  
    * Create a folder (for example, /home/dooseop/DATASET/) and copy the driving sequences (link) listed in './dataset/train.txt' and './dataset/test.txt' into the created folder.    
    * Copy 'preprocessed_dataset.cpkl' (https://www.dropbox.com/sh/ymj579oowpqlk0y/AABzOccIgaItbpIRNu99XOrea?dl=0) into './dataset/'.  

2) Pretrained Network Params  
    * Create a folder './pretrained_cnn' and copy 'saved_cnn_exp12_model70.pt' (https://www.dropbox.com/sh/mdph47lt8l3kw8w/AADI1xVqf6uMCznwTcS-cCLFa?dl=0) into the created folder.  

3) Libraries
    * python 3.6  
    * pytorch 1.1.0  
    * torchvision 0.3.0  
    * opencv 3.1 >=  
    * scipy 1.3.0 >=  
    * scikit-image 0.14 >=
    * scikit-learn 0.22 >=  
  
## Train New Models
To train the model from scratch, run the following. The parameters of the trained networks will be stored in './saved_models/model100'.
```sh
$ python train.py --exp_id 100
```

To evaluate the trained model, run the following.
```sh
$ python eval.py --exp_id 100
```

To draw 300 paths generated by the pretrained model (for example, 'saved_chk_point_95.pt') on the frontview images, run the following.
```sh
$ python visualization.py --exp_id 100 --model_num 95 --besk_k 300
```

## Pretrained Models
Download the pre-trained models from (https://www.dropbox.com/sh/7e17hoeom58x54d/AABjVUlTj67KQ55Whe40xEcQa?dl=0) and copy them into './saved_models/'.  

## Citation
```
@article{Choi,
 author = {D. Choi and S.-J. Han and K. Min and J. Choi},
 title = {PathGAN: Local Path Planning with Attentive Generative Adversarial Networks},
 journal = {ETRI Journal},
 doi= {https://doi.org/10.4218/etrij.2021-0192},
 year = {2022}
}
```
