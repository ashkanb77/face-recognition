from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import logging

logging.getLogger().setLevel(logging.INFO) # show reports in info level
logger = logging.getLogger('Face-Model')


class AverageMeter:
  
  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0


  def update(self, val, n=1):
    self.val = val
    self.sum = val * n
    self.count += n
    self.avg = self.sum / self.count


class Checkpoint:

  def __init__(self, folder, model):
    self.best_acc = 0.
    self.folder = folder
    self.model = model
    os.makedirs(self.folder, exist_ok=True)

  def save(self, acc, file_name, epoch=-1):
    if acc > self.best_acc:
      logger.info('model checkpoint...')
      state = {
          'model': self.model.state_dict(),
          'acc': acc,
          'epoch': epoch
      }
      path = os.path.join(os.path.abspath(self.folder), file_name+ '.pth')
      torch.save(state, path)
      self.best_acc = acc

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])

    return train_transform

def plot_images(images, labels, classes):
  n_images = len(images)

  rows = int(np.sqrt(n_images))
  cols = int(np.sqrt(n_images))

  fig = plt.figure(figsize=(20, 20))

  for i in range(rows*cols):

    ax = fig.add_subplot(rows, cols, i+1)
    image = images[i]

    image = image * 128 + 127.5
    image = image.type(torch.int16)

    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax.set_title(classes[labels[i]])
    ax.axis('off')

def count_params(model):
  params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return params
