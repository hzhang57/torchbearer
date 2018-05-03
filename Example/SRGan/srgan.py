"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import torch
from bink import Model
from bink.metrics import Mean, RunningMean, Metric
from torch.utils.data import DataLoader

from LossCallbacks import *
from datasets import *
from models import *

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

epoch = 0
n_epochs = 200
dataset_name = 'img_align_celeba'
batch_size = 1
lr = 0.0002
b1 = 0.5
b2 = 0.999
decay_epoch = 100
n_cpu = 8
hr_height = 256 # 'size of high res. image height'
hr_width = 256
channels = 3
sample_interval = 100 # 'interval between sampling of images from generators'
checkpoint_interval = -1

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(hr_height / 2**4), int(hr_width / 2**4)
patch = (batch_size, 1, patch_h, patch_w)

torchmodel = GANModel()

# Initialize generator and discriminator
generator = torchmodel.generator
discriminator = torchmodel.discriminator
feature_extractor = torchmodel.feature_extractor

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth'))
    discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth'))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optim_g = torch.optim.Adam([{'params': generator.parameters()}, {'params': discriminator.parameters()}]
                           , lr=lr, betas=(b1,b2))

# Transforms for low resolution images and high resolution images
lr_transforms = [   transforms.Resize((hr_height//4, hr_height//4), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

hr_transforms = [   transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(ImageDataset("/home/matt/PycharmProjects/PyTorch-GAN/data/%s" % dataset_name, lr_transforms=lr_transforms, hr_transforms=hr_transforms),
                        batch_size=batch_size, shuffle=True, num_workers=n_cpu)


def generator_loss(outputs, imgs_hr):
    return Variable(torch.zeros(1))

class LossDMetric(Metric):
    def process(self, state):
        super().process(state)
        return state['loss_D'].data

class LossGMetric(Metric):
    def process(self, state):
        super().process(state)
        return state['loss'].data

loss_g_m = LossGMetric('loss_g')
loss_d_m = LossDMetric('loss_d')


model = Model(torchmodel, optim_g, generator_loss, [RunningMean(loss_d_m), RunningMean(loss_g_m), Mean(loss_g_m), Mean(loss_d_m)]).cuda()
model.fit_generator(dataloader, epochs=5, callbacks=[GenLoss(), DetLoss()], pass_state=True)

