# OpenPose

## Preparation
1. cài conda
    - https://www.anaconda.com/products/individual
    
2. Create environment for project
    - conda create -n Talent5OpenPose python=3.8
    
3. Install pytorch
    - cpu  
      - conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cpuonly -c pytorch
   - gpu
      - pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

   
4. Other library
   - cd src\pose_estimation
   - pip install -r requirements.txt
   
## Create video to detect person's posture in image or video
1. cd pose_estimation
2. python open_pose_video.py
