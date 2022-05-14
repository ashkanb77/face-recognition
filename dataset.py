from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from facenet_pytorch import MTCNN

class LFW(Dataset):

  def __init__(self, path, image_size, transform=None):
    self.path = path
    self.image_size = image_size
    self.dataset = ImageFolder(path, transform=transform)
    self.transform = transform
    self.mtcnn = MTCNN(image_size=image_size)


  def __getitem__(self, item):
    img_path, label = self.dataset.imgs[item]
    img = Image.open(img_path)
    img_cropped = self.mtcnn(img)
    if img_cropped == None:
      img = img.resize((self.image_size, self.image_size))
      img_cropped = transforms.ToTensor()(img)
      img_cropped = img_cropped * 2 - 127.5/128.0

    self.dataset.transform(img_cropped)
    return img_cropped, label

  def __len__(self):
    return len(self.dataset)