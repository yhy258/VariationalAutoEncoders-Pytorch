# VariationalAutoEncoders-Pytorch
  
## Standard VAE
paper : https://arxiv.org/abs/1312.6114  
Model : https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/standard_vae.py
  
**설명** :
https://deepseow.tistory.com/38  
https://deepseow.tistory.com/39  
https://deepseow.tistory.com/40  
  
**후기, Comment** :  
개인적인 경험에 의하면 Reconstruction Loss 작성 시 Tanh + MSELoss보다 Sigmoid + BCELoss가 더 작동을 잘했다.  
단 설정 상 달라야 하는 점은 Reconstruction Loss를 사용할 때 Reconstruction Loss와 Regulerization Term 사이의 균형에 관한 weight를 부여해줘야 한다는 점.  
훈련을 동향을 살펴보고 하면 될 듯. 기본적으로 Regularization Term (KLD) 앞에 weight로 Batch Size/Dataset Size를 곱해주면 대충 돌아가긴했다. 하지만 BCE보다 결과가 살짝 안좋았음.(크게 차이는 없지만,)  
그리고 Posterior Collapse가 일어나곤한다.  

In my experience, Sigmoid + BCELoss works better than Tanh + MSELoss when using Reconstruction Loss.  
The only difference is that when using Reconstruction Loss, you must balance Reconstruction Loss and Regulation Term.  
You can do this by looking at your training trends. Basically, I multiplied the Regularization Term (KLD) by the Batch Size/Dataset Size with the weight, and it worked. However, the results were slightly worse than those of BCE (though not by much).  
And it tends to cause Posterior Collapse.  

  
**실험 결과, Results**  
![Reconstruction](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/Standard_VAE_bce2_Reconstruction.png?raw=true)
![Sampling](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/Standard_VAE_bce2_Sampling.png?raw=true)  
Reconstruction 우측, Sampling 좌측. 100epoch 실험. Sampling이 다소 흐리다.  
이 결과는 훈련이 다소 부족했는듯. 자원이 부족해서 돌리기가 힘들었다. 그리고 GAN보다는 확실히 좀 blur한 결과.

Reconstructed samples are on the right, and Sampled samples are on the left. 100 epoch experiment. Sampled results are somewhat blurry.  
This result might be caused by a lack of training. It was hard to run due to a lack of resources, and the result is definitely a bit blurrier than GAN.
  
  
## Beta VAE
paper : https://openreview.net/forum?id=Sy2fzU9gl, https://arxiv.org/abs/1804.03599  
Model : https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/beta_VAE.py  
Analyze : https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/beta_vae_analyze.py  
  
**후기** **Comment**:  
첫 beta vae가 제안된 논문을 읽어보면 뭔가 불분명했지만, Understanding paper를 읽은 후 InfoGAN과 비슷하게 상호정보량을 통해 어느정도 이해가 되었고 이를 기반으로 더 나은 방법인 weight를 linear하게 높이는 방법을 알게 되었다.  
이렇게 단순히 KL Divergence Term에 weight를 붙이는 것 만으로도 Disentangle하게 할 수 있다니.. 하지만 논문을 읽어도 이는 다소 휴리스틱하게 느껴졌다.  

When I first read the paper where beta-vae was proposed, it was unclear, but after reading the Understanding paper, I understood it somewhat through mutual information quantity similar to InfoGAN, and based on that, I realized a better method, how to increase the weight linearly.  
Amazingly, we can disentangle the KL Divergence Term by simply adding weight to it.
  
**실험 결과, Results**  
![Reconstruction](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/latent32_beta_vae_recons.png?raw=true) 
![Sampling](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/latent32_beta_vae_sampling.png?raw=true)  
좌측 이미지 Reconstruction, 우측 이미지 Sampling  
데이터 셋의 크기가 커서 64 batch size, 30 epoch에도 많은 iteration 돌았음. 그리고 개인적인 생각으로 CelebA dataset에 대한 모델을 다소 간단하게 구성해서 blur가 더 심하지 않나 라는 생각을 한다.  
blur에 대한 이유는 이 뿐만 아니라 논문에서도 나와있듯이, latent vector의 size가 어떻냐에도 달려있다.  

Reconstructed samples are on the right, and Sampled samples are on the left.
Due to the large size of the dataset, it took many iterations even with 64 batch size and 30 epochs. And personally, I think that the model for CelebA dataset is somewhat simpler, so the blur is worse.  
The reason for the blurred results is not only this but also depends on the size of the latent vector, as shown in the paper.  

![Reconstruction](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/beta_VAE_latent_10_reconstruction.png?raw=true)
![Sampling](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/beta_VAE_latent_10_Sampling.png?raw=true)  
이 결과는 10 latent size를 갖는 경우의 결과이다. 위와 동일하게 30epoch 정도 훈련 시켰다. 비교적 선명해보인다.  
This result is for a latent size of 10. We trained the same as the above for about 30 epochs. It looks relatively sharp.  


  
**분석, Analysis**  
이 부분은 representation 분석 부분이다.  
![BetaAnalyze](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/smilewoman.png?raw=true)  
Analyze는 간단하게 진행했다. 웃는 여자, 웃지 않는 여자 사진에 대한 latent를 빼서 smile latent 구성 (average)  
이를 smile factor로 두고 linear하게 factor 크기를 높여주면서 이미지 변화 확인  
We simply performed the analysis. We construct the latent of the smile factor by subtracting the latent vectors for the smiling and non-smiling woman photos.
We set it to the smile factor and increased the factor size linearly to see how the image changed.


## VQ(Vector Quantization) VAE
paper : https://arxiv.org/pdf/1711.00937.pdf  
Model : https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/vq_vae.py  

**설명** :  
https://deepseow.tistory.com/41  
  
**후기, Comment** :  
논문에 Vector Quantization 기법이 뭔지 자세히 안나와있어서 이해하는데 꽤나 고생했다. 결국 미리 짜여져 있는 코드를 기반으로 공부해나가면서 코드를 짰다.  
지금까지 딥러닝을 사용한 컴퓨터 비전을 공부해오면서 Vector Quantization이라는 개념을 한번도 본 적이 없었는데, 새로운 개념을 배우게 되어서 성장한 기분이 들었다 :)  
레퍼런스 한 깃허브 : https://github.com/zalandoresearch/pytorch-vq-vae  
레퍼런스 했기 때문에 위 코드와 상당히 비슷. 그리고 EMA 기법에 있어서 laplace smoothing 과 같은 기법이 추가로 사용되었다.  
주의해야 할 점 : 본 코드는 PixelCNN을 이용한 autoregressive 형태의 샘플링은 제외했다.  
샘플링 시 훈련된 Embedding space에 대해 PixelCNN을 Fitting하고 해당 PixelCNN 모델을 통해 latent를 모델링 하여 latent 가지고 이미지를 생성하면된다.  
여기에서 의문이 들었던 점은 Embedding space에 대해 PixelCNN을 어떻게 fitting 하느냐였는데, 열심히 찾아서 봐본 결과 우선, VQVAE를 훈련시켰던 데이터셋을 가지고 encoding을 통해 latents를 뽑아 낸 후 해당 Latents를 가지고 PixelCNN을 훈련시킨다. 이후 훈련 PixelCNN을 가지고 Autoregressive한 과정을 통해 어떤 feature map (latents) 만들어 낸 후 이걸 가지고 decoder로 넣어서 샘플링한다.

I had a hard time understanding the Vector Quantization technique because it was not explained in detail in the paper, so I ended up writing code based on the pre-written code I studied.  
I had never come across the concept of Vector Quantization before while studying computer vision using deep learning, but I felt like I grew by learning a new concept :)  
I referenced GitHub : https://github.com/zalandoresearch/pytorch-vq-vae  
It is quite similar to the code above. Additional techniques, such as place smoothing, are also used in the EMA technique.  
Note: This code excludes autoregressive sampling using PixelCNN.  
When sampling, you can fit a PixelCNN to the trained embedding space and model the latent through the PixelCNN model to generate an image with the latent.  
The question here was how to fit PixelCNN to the embedding space, but after searching hard, I found that first, we take the dataset that trained VQVAE, extract latent through encoding, and train PixelCNN with those patients. After that, we create feature maps (latent vectors) through an autoregressive process with the training PixelCNN and then put it into the decoder to sample.

  
**실험 결과, Results**  
![Reconstruction](https://github.com/yhy258/VariationalAutoEncoders-Pytorch/blob/master/Images/VQ_VAE_Reconstruction.png?raw=true)  
위 사진은 VQ-VAE를 이용한 Reconstruction 결과이다. vae 결과들보다 더 좋은 결과를 내놓음을 한눈에 알 수 있었다.  

Above is the reconstruction using VQ-VAE. Its results are better than those of other VAE models.  

