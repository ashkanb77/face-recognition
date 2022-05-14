import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import get_transforms, plot_images, count_params, AverageMeter, Checkpoint
from dataset import LFW
from model import Model
import torchmetrics


logging.getLogger().setLevel(logging.INFO) # show reports in info level
logger = logging.getLogger('Face-Model')

parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs', type=int, default=100, required=True, help='number of epochs for training')
parser.add_argument('--base_dir', type=str, required=True,
           help='base directory of dataset that contains train and test folder')
parser.add_argument('--plot_images', type=bool, default=False, help='plot one batch of images')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--experiment', type=str, default='experiment1', help='experiment path')
parser.add_argument('--image_size', type=int, default=72, help='experiment path')


args = parser.parse_args()

writer = SummaryWriter('runs/' + args.experiment)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

lfw_train = LFW(args.base_dir + 'train', args.image_size, transform=get_transforms())
train_loader = DataLoader(lfw_train, batch_size=args.batch_size, shuffle=True)

lfw_test = LFW(args.base_dir + 'test', transform=None)
test_loader = DataLoader(lfw_test, batch_size=args.batch_size, shuffle=True)

classes = lfw_train.dataset.classes
N_CLASSES = len(classes)

if args.plot_images:
    batch = next(iter(train_loader))
    images, labels = batch

    plot_images(images, labels, classes)


model = Model()
model.to(device)

print(f"number of parameters of model is {count_params(model)}")

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), args.learning_rate)

def train(epoch):
  model.train()
  loss_total = AverageMeter()
  accuracy = torchmetrics.Accuracy().cuda()
  for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    loss_total.update(loss)
    accuracy(outputs.softmax(dim=-1), targets)

    # if batch_idx == 5:
    #   break

  acc = accuracy.compute()
  writer.add_scalar('Loss/train', loss_total.avg.item(), epoch)
  writer.add_scalar('Acc/train', acc.item(), epoch)
  print(f"train: Epoch: {epoch}, Loss: {loss_total.avg:.4} Accuracy: {acc:.4}")


def test(epoch, checkpoint):
  model.eval()
  loss_total = AverageMeter()
  accuracy = torchmetrics.Accuracy().cuda()
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss_total.update(loss)
      accuracy(outputs.softmax(dim=-1), targets)

    acc = accuracy.compute()
    writer.add_scalar('Loss/test', loss_total.avg.item(), epoch)
    writer.add_scalar('Acc/test', acc.item(), epoch)
    print(f"test: Epoch: {epoch}, Loss: {loss_total.avg:.4} Accuracy: {acc:.4}")
    print()

  checkpoint.save(accuracy.compute(), 'ckpt', epoch=epoch)


checkpoint = Checkpoint()
for epoch in range(args.n_epochs):
  train(epoch)
  test(epoch, checkpoint)
