import numpy as np

from bink.callbacks import Callback
from torch.autograd import Variable
import torch


class GenLoss(Callback):
    def on_start(self, state):
        super().on_start(state)
        state['criterion_GAN'] = torch.nn.MSELoss()
        state['criterion_content'] = torch.nn.L1Loss()
        # state['optim_d'] = torch.optim.Adam(state['model'].discriminator.parameters(), lr= 0.0002, betas=(0.5, 0.999))

        state['feature_extractor'] = state['model'].feature_extractor

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor

        # Calculate output of image discriminator (PatchGAN)
        patch_h, patch_w = int(256 / 2 ** 4), int(256 / 2 ** 4)
        patch = (1, 1, patch_h, patch_w)

        # Adversarial ground truths
        state['valid'] = Variable(Tensor(np.ones(patch)), requires_grad=False)
        state['fake'] = Variable(Tensor(np.zeros(patch)), requires_grad=False)

    def on_forward_criterion(self, state):
        super().on_forward_criterion(state)
        loss_GAN = state['criterion_GAN'](state['gen_validity'], state['valid'])
        gen_features = state['feature_extractor'](state['gen_hr'])
        real_features = Variable(state['feature_extractor'](state['y_true']).data, requires_grad=False)
        loss_content = state['criterion_content'](gen_features, real_features)

        state['loss'] = 1e-3*loss_GAN + loss_content


class DetLoss(Callback):
    def on_sample(self, state):
        super().on_sample(state)
        # state['optim_d'].zero_grad()

    def on_forward_criterion(self, state):
        super().on_forward_criterion(state)
        # Loss
        loss_real = state['criterion_GAN'](state['desc_imgs_hr'], state['valid'])
        loss_fake = state['criterion_GAN'](state['desc_gen_hr'],  state['fake'])

        state['loss_D'] = (loss_real + loss_fake) / 2

    def on_backward(self, state):
        super().on_backward_criterion(state)
        state['loss_D'].backward()
        # state['optim_d'].step()
