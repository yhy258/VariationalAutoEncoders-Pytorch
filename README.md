# VariationalAutoEncoders-Pytorch
  
## Standard VAE
Model : https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/standard_vae.py
  
설명 :
https://deepseow.tistory.com/38  
https://deepseow.tistory.com/39  
  
후기 :  
개인적인 경험에 의하면 Reconstruction Loss 작성 시 Tanh + MSELoss보다 Sigmoid + BCELoss가 더 작동을 잘했다.  
단 설정 상 달라야 하는 점은 Reconstruction Loss를 사용할 때 Reconstruction Loss와 Regulerization Term 사이의 균형에 관한 weight를 부여해줘야 한다는 점.  
훈련을 동향을 살펴보고 하면 될 듯. 기본적으로 Regularization Term (KLD) 앞에 weight로 Batch Size/Dataset Size를 곱해주면 대충 돌악가긴했다. 하지만 BCE보다 결과가 살짝 안좋았음.  
