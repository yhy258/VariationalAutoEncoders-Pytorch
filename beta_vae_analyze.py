import torch

from torchvision import transforms
from torchvision.transforms.functional import crop
from torchvision.utils import save_image


from beta_VAE import betaVAE

import numpy as np
from PIL import Image


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae_model = betaVAE(max_iter=1e5, C_value=25, input_shape=(3, 64, 64), beta=1000).to(DEVICE)
vae_model.load_state_dict(torch.load("/content/drive/MyDrive/model_save/betaVAE"))

woman_smiles_ids = ['000145.jpg', '000149.jpg', '000151.jpg']
woman_ids = ['000157.jpg', '000190.jpg']

man_ids = ['000008.jpg', '000055.jpg', '000065.jpg']

data_root = ''
# # Path to folder with the dataset
dataset_folder = f'{data_root}/img_align_celeba'
FOLDER_PATH = f'{dataset_folder}/img_align_celeba/'

im_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])


# returns pytorch tensor of images
def get_imgs(name_list):
    ims = []
    for im_id in name_list:
        im_path = FOLDER_PATH + im_id
        im = Image.open(im_path)
        im = crop(im, 30, 0, 178, 178)
        im = im_transform(im)
        ims.append(im)
    return torch.cat(ims, dim=0).view(-1, *im.size())


def get_attr_latent(imgs, model, DEVICE):
    imgs = imgs.to(DEVICE)
    mu, log_var = model.encoder(imgs)
    return model.reparameterize_trick(mu, log_var)


def average_latent(latent):  # latent shape = [BS, latent_dim]
    return torch.mean(latent, dim=0)


def latent_arithmetic(im_latent, attr_latent, model, DEVICE):
    factors = np.linspace(0, 1.5, num=10, dtype=float)
    result = []

    with torch.no_grad():
        for f in factors:  # 해당 attribute에 대한 factor를 서서히 넣어줘서 결과를 봐보자.
            z = im_latent + (f * attr_latent).type(torch.FloatTensor).to(DEVICE)

            im = torch.squeeze(model.decoder(z.unsqueeze(0)).cpu())
            result.append(im)

    return result


# latent 상에서 특징당 averge해서 빼자. ex) sunglass쓴 man average - man average
if __name__ == '__main__':
    woman_smiles_imgs = get_imgs(woman_smiles_ids)
    man_imgs = get_imgs(man_ids)
    woman_imgs = get_imgs(woman_ids)

    man_latent = get_attr_latent(man_imgs[0].unsqueeze(0), vae_model, DEVICE)
    woman_latent = get_attr_latent(woman_imgs[1].unsqueeze(0), vae_model, DEVICE)

    woman_avg_latent = average_latent(get_attr_latent(woman_imgs, vae_model, DEVICE))
    woman_smiles_latent = average_latent(get_attr_latent(woman_smiles_imgs, vae_model, DEVICE))

    smile_latent = woman_smiles_latent - woman_avg_latent

    smile_man_results = latent_arithmetic(man_latent, smile_latent, vae_model, DEVICE)
    smile_woman_results = latent_arithmetic(woman_latent, smile_latent, vae_model, DEVICE)

    save_image(smile_woman_results, "/smile_woman.png", nrow=5, normalize=False)
    save_image(smile_man_results, "/smile_man.png", nrow=5, normalize=False)





