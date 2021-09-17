import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

from vae_model_frame import *
from data_utils import CelebADataset
import numpy as np

from tqdm import tqdm


# Simple Implementation
class betaVAE(VAE_Frame):
    def __init__(self, max_iter, C_value, input_shape=(3, 32, 32), beta=1):
        super().__init__()
        self.input_shape = input_shape
        self.beta = beta
        self.max_iter = max_iter
        self.C_value = torch.tensor(C_value, requires_grad=False)

        self.iter = 0

        img_in_channels = input_shape[0]

        if input_shape[1] == 32:
            hid_dims = [32, 64, 64]
            latent_dim = 32
            self.start_width = 4

        elif input_shape[1] == 64:
            hid_dims = [32, 32, 64, 64]
            latent_dim = 32
            self.start_width = 4

        else:
            raise NotImplementedError("input_shape parameter only 32, 64")

        self.latent_dim = latent_dim

        # Encoder
        self.encoder_modules = []

        in_channels = img_in_channels

        for h_dim in hid_dims:
            self.encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder_modules = nn.Sequential(*self.encoder_modules)
        self.encoder_fc = nn.Linear(hid_dims[-1] * self.start_width * self.start_width, 256)
        self.encode_mu = nn.Linear(256, latent_dim)
        self.encode_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.proj_decode = nn.Linear(latent_dim, hid_dims[-1] * self.start_width * self.start_width)

        self.decoder_modules = []
        hid_dims.reverse()

        for i in range(len(hid_dims) - 1):
            self.decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hid_dims[i],
                                       hid_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hid_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hid_dims[-1],
                                   hid_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hid_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hid_dims[-1], out_channels=img_in_channels,
                          kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder_modules = nn.Sequential(*self.decoder_modules)

    def encoder(self, x: Tensor) -> List[Tensor]:
        x = self.encoder_modules(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        mu = self.encode_mu(x)
        log_var = self.encode_logvar(x)
        return [mu, log_var]

    def decoder(self, z: Tensor) -> Tensor:
        z = self.proj_decode(z)
        z = z.view(z.size(0), -1, self.start_width, self.start_width)
        return self.decoder_modules(z)

    def reparameterize_trick(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return std * eps + mu

    def generate(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    def sample(self, samples_num: input, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> List[
        Tensor]:
        latent = torch.randn(samples_num, self.latent_dim).to(device)
        return self.decoder(latent)

    def forward(self, x: Tensor) -> List[Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize_trick(mu, log_var)
        output = self.decoder(z)
        return output, mu, log_var

    def loss_function(self, pred: Tensor, real_img: Tensor, mu: Tensor, log_var: Tensor):
        # VAE 논문 마지막 부분에 나와있음. (ELBO를 수식적으로 유도한 결과)
        self.iter += 1
        C = torch.clamp(self.C_value / self.max_iter * self.iter, min=0, max=self.C_value.item())

        recons_loss = nn.BCELoss(size_average=False)(pred, real_img) / pred.size(0)
        kld_loss = ((mu ** 2 + log_var.exp() - 1 - log_var) / 2).mean()
        loss = recons_loss + self.beta * (self.latent_dim / pred.size(0)) * torch.abs(kld_loss - C)

        return loss, recons_loss, kld_loss

# Simple Implementation
if __name__ == "__main__":
    ngpu = 1
    device = torch.device('cuda:0' if (
            torch.cuda.is_available() and ngpu > 0) else 'cpu')

    img_folder = ""

    transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
            ])

    celeba_dataset = CelebADataset(img_folder = "", transform = transform)

    ## Create a dataloader
    # Batch size during training
    batch_size = 64
    # Number of workers for the dataloader
    num_workers = 2 if device.type == 'cuda' else 2
    # Whether to put fetched data tensors to pinned memory
    pin_memory = True if device.type == 'cuda' else False

    dataloader = torch.utils.data.DataLoader(celeba_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             shuffle=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae_model = betaVAE(max_iter=1e5, C_value=25, input_shape=(3, 64, 64), beta=1000).to(DEVICE)
    optim = torch.optim.Adam(params=vae_model.parameters(), lr=1e-3)

    # vae_model.load_state_dict(torch.load("/content/drive/MyDrive/model_save/betaVAE"))

    for epoch in range(30):  # 10000
        print("Epoch : {}/30".format(epoch + 1))
        losses = []
        recons_losses = []
        kld_losses = []

        for img in tqdm(dataloader):
            img = img.to(DEVICE)
            pred, mu, log_var = vae_model(img)
            total_loss, recons_loss, kld_loss = vae_model.loss_function(pred, img, mu, log_var)

            losses.append(total_loss.item())
            recons_losses.append(recons_loss.item())
            kld_losses.append(kld_loss.item())

            optim.zero_grad()
            total_loss.backward()
            optim.step()

        print("Total Losses : {:.5f} Reconstruction Losses : {:.5f} KL Divergence : {:.5f}".format(np.mean(losses),
                                                                                                   np.mean(
                                                                                                       recons_losses),
                                                                                                   np.mean(kld_losses)))

        torch.save(vae_model.state_dict(), "betaVAE_model.pt")

    """
        Reconstruction
    """
    imgs = next(iter(dataloader))
    imgs = imgs.to(DEVICE)
    recons_datas = vae_model.generate(imgs)
    print(recons_datas.size())
    save_image(recons_datas.view(batch_size, 3, 64, 64)[:25],
               "_reconstruction.png", nrow=5, normalize=False)

    """
        Sampling
    """
    samples = vae_model.sample(25)
    save_image(samples, "_Sampling.png", nrow=5, normalize=False)






