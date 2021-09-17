import os
import zipfile
import gdown
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def download_CelebA(data_root):

    dataset_folder = f'{data_root}/img_align_celeba'
    # URL for the CelebA dataset
    url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
    # Path to download the dataset to
    download_path = f'{data_root}/img_align_celeba.zip'

    # Create required directories
    if not os.path.exists(data_root):
        os.makedirs(data_root)
        os.makedirs(dataset_folder)

    gdown.download(url, download_path, quiet=False)

    # Unzip the downloaded file
    with zipfile.ZipFile(download_path, 'r') as ziphandler:
        ziphandler.extractall(dataset_folder)

class CelebADataset(Dataset):
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform
    self.image_names = natsorted(image_names)

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    return img


