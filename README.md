#### This is the demo code of our paper "A Simple Adaptive Unfolding Network for Hyperspectral Image Reconstruction" in submission to IJCAI 2023.

This repo includes:  

- Specification of dependencies.
- Training code.
- Evaluation code.
- Geting params and FLOPs code.
- Testing training time and inference speed code.
- README file.

This repo can reproduce the main results in Tabel (1) and Tabel (2) of our main paper.
All the source code and pre-trained models will be released to the public for further research.


#### 1. Create Environment:

------
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- [PyTorch >= 1.3](https://pytorch.org/)

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```

#### 2. Prepare Dataset:

Download the dataset from https://github.com/mengziyi64/TSA-Net, put the dataset into the corresponding folder 'code/datasets/', and recollect them in the following form:

    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
#### 3. Training and Testing for simulation experiment:
##### Training 
    #SAUNet-1stg
    python simu_train.py --method saunet_1stg --outf ./exp/simu_saunet_1stg/ --seed 24 --gpu_id 0
    #SAUNet-2stg
    python simu_train.py --method saunet_2stg --outf ./exp/simu_saunet_2stg/ --seed 24 --gpu_id 0
    #SAUNet-3stg
    python simu_train.py --method saunet_3stg --outf ./exp/simu_saunet_3stg/ --seed 24 --gpu_id 0
    #SAUNet-5stg
    python simu_train.py --method saunet_5stg --outf ./exp/simu_saunet_5stg/ --seed 24 --gpu_id 0
    #SAUNet-9stg 
    python simu_train.py --method saunet_9stg --outf ./exp/simu_saunet_9stg/ --seed 24 --gpu_id 0 
    #SAUNet-13stg 
    python simu_train.py --method saunet_13stg --outf ./exp/simu_saunet_13stg/ --seed 24 --gpu_id 0 

    Please use checkpointing (--cp) when running out of memory. refer to 'utils/simu_utils/simu_args.py' to use more options.
##### Testing 
a). Test our models on the HSI dataset. The results will be saved in 'code/evaluation/testing_result/' in the MatFile format. For example, we test the SAUNet-3stg:

    python simu_test.py --method saunet_3stg --outf ./test/simu_saunet_3stg  --pretrained_model_path [your saunet_3stg model path]

b). Calculate quality assessment. We use the same quality assessment code as DGSMP. So please use Matlab, get in 'code/analysis_tools/Quality_Metrics/', and then run 'Cal_quality_assessment.m'.

c). If you want test SAUNet-1stg or the others , please change the model your want to test in above step a).

#### 4. Training and Testing for real data experiment:
##### Training 
    #SAUNet-1stg
    python real_train.py --method saunet_1stg --outf ./exp/real_saunet_1stg/ --seed 24 --gpu_id 0 --isTrain
    #SAUNet-2stg
    python real_train.py --method saunet_2stg --outf ./exp/real_saunet_2stg/ --seed 24 --gpu_id 0 --isTrain
    #SAUNet-3stg
    python real_train.py --method saunet_3stg --outf ./exp/real_saunet_3stg/ --seed 24 --gpu_id 0 --isTrain
    #SAUNet-5stg
    python real_train.py --method saunet_5stg --outf ./exp/real_saunet_5stg/ --seed 24 --gpu_id 0 --isTrain
    #SAUNet-9stg 
    python real_train.py --method saunet_9stg --outf ./exp/real_saunet_9stg/ --seed 24 --gpu_id 0 --isTrain
    #SAUNet-13stg 
    python real_train.py --method saunet_13stg --outf ./exp/real_saunet_13stg/ --seed 24 --gpu_id 0 --isTrain

    Please use checkpointing (--cp) when running out of memory.
##### Testing 
a). Test our models on the HSI dataset. The results will be saved in 'code/evaluation/testing_result/' in the MatFile format. For example, we test the SAUNet-3stg:

    python real_test.py --method saunet_3stg --outf ./test/real_saunet_3stg/ --pretrained_model_path [your saunet_3stg model path]

b). Calculate quality assessment. We use no reference image quality assessments (Naturalness Image Quality Evaluator, **NIQE** ). So please use Matlab, get in 'code/analysis_tools/Quality_Metrics/', and then run 'NIQE_metric.m'.

c). If you want test SAUNet-1stg or the others , please change the model your want to test in above step a).

#### 5. Get training time and inference FPS
##### Inference FPS
If we want to get inference fps of SAUNet-3stg, run the following commond:

    python test_fps.py --method saunet_3stg --outf ./test/real_saunet_3stg --gpu_id 0
**Please mask sure that the GPU is not occupied by another program before running the commond.** Other models are similar to this.

##### Training time
Afer you finish the training of model, please run these commands:

    cd analysis_tools/
    python tranining_time [your training log path]

#### 6. Evaluating the Params and FLOPS of models
You can get the Params and FLOPS of models **at the begin of training**. Or use following commonds 
(for instance, we get these values of SAUNet-3stg. Other methods are similar):

    python test_fps.py --method saunet_3stg --outf [your log path to save]

#### 7. This repo is mainly based on *the toolbox for Spectral Compressive Imaging*, which is provided by MST and contains 11 learning-based algorithms for spectral compressive imaging. 
The above toolbox offer us a fair benchmark comparison. We use the methods correspoding to original repo as follows:

(1)  TSA-Net: https://github.com/mengziyi64/TSA-Net

(2)  DGSMP: https://github.com/TaoHuang95/DGSMP

(3) GAP-Net: https://github.com/mengziyi64/GAP-net

(4) ADMM-Net: https://github.com/mengziyi64/ADMM-net

(5) HDNet: https://github.com/Huxiaowan/HDNet

(6) MST: https://github.com/caiyuanhao1998/MST

(7) CST: https://github.com/caiyuanhao1998/MST

(8) DAUHST: https://github.com/caiyuanhao1998/MST

We thank these repos and have cited these works in our manuscript.