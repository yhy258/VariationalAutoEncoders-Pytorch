# VariationalAutoEncoders-Pytorch
  
## Standard VAE
paper : https://arxiv.org/abs/1312.6114  
Model : https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/standard_vae.py
  
**설명** :
https://deepseow.tistory.com/38  
https://deepseow.tistory.com/39  
https://deepseow.tistory.com/40  
  
**후기** :  
개인적인 경험에 의하면 Reconstruction Loss 작성 시 Tanh + MSELoss보다 Sigmoid + BCELoss가 더 작동을 잘했다.  
단 설정 상 달라야 하는 점은 Reconstruction Loss를 사용할 때 Reconstruction Loss와 Regulerization Term 사이의 균형에 관한 weight를 부여해줘야 한다는 점.  
훈련을 동향을 살펴보고 하면 될 듯. 기본적으로 Regularization Term (KLD) 앞에 weight로 Batch Size/Dataset Size를 곱해주면 대충 돌아가긴했다. 하지만 BCE보다 결과가 살짝 안좋았음.(크게 차이는 없지만,)  
그리고 Posterior Collapse가 일어나곤한다.  
  
**실험 결과**  
![Reconstruction](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/Standard_VAE_bce2_Reconstruction.png?raw=true)  
![Sampling](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/Standard_VAE_bce2_Sampling.png?raw=true)
Reconstruction 우측, Sampling 좌측. 100epoch 실험. Sampling이 다소 흐리다.  
이 결과는 훈련이 다소 부족했는듯. 학부생이라 자원이 부족해서 돌리기가 힘들다. 그리고 GAN보다는 확실히 좀 blur한 결과.
  
  
## Beta VAE
paper : https://openreview.net/forum?id=Sy2fzU9gl, https://arxiv.org/abs/1804.03599  
Model : https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/beta_VAE.py  
Analyze : https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/beta_vae_analyze.py  
  
**후기** :  
첫 beta vae가 제안된 논문을 읽어보면 뭔가 불분명했지만, Understanding paper를 읽은 후 InfoGAN과 비슷하게 상호정보량을 통해 어느정도 이해가 되었고 이를 기반으로 더 나은 방법인 weight를 linear하게 높이는 방법을 알게 되었다.  
이렇게 단순히 KL Divergence Term에 weight를 붙이는 것 만으로도 Disentangle하게 할 수 있다니.. 하지만 논문을 읽어도 이는 다소 휴리스틱하게 느껴졌다.  
  
**실험 결과**  
![Reconstruction](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/latent32_beta_vae_recons.png?raw=true) 
![Sampling](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/latent32_beta_vae_sampling.png?raw=true)  
좌측 이미지 Reconstruction, 우측 이미지 Sampling  
데이터 셋의 크기가 커서 30 epoch에도 많은 iteration 돌았음. 그리고 개인적인 생각으로 CelebA dataset에 대한 모델을 다소 간단하게 구성해서 blur가 더 심하지 않나 라는 생각을 한다.  
blur에 대한 이유는 이 뿐만 아니라 논문에서도 나와있듯이, latent vector의 size가 어떻냐에도 달려있다.  
![Reconstruction](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/beta_VAE_latent_10_reconstruction.png?raw=true)
![Sampling](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/beta_VAE_latent_10_Sampling.png?raw=true)  
이 결과는 10 latent size를 갖는 경우의 결과이다. 위와 동일하게 30epoch 정도 훈련 시켰다. 비교적 선명해보인다.  


  
**분석**  
이 부분은 disentagle의 분석 부분이다.  
![BetaAnalyze](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/smilewoman.png?raw=true)  
Analyze는 간단하게 진행했다. 웃는 여자, 웃지 않는 여자 사진에 대한 latent를 빼서 smile latent 구성 (average)  
이를 smile factor로 두고 linear하게 factor 크기를 높여주면서 이미지 변화 확인  
disentangle!  
