import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import DepthsepCCBlock as DepthsepCCBlock
import pdb

class CondConvGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if opt.dataset_mode == 'cityscapes':
            self.num_upsampling_layers = 'more'
        else:
            self.num_upsampling_layers = 'normal'

        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16*nf*self.sw*self.sh)            
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16*nf*self.sw*self.sh, 3, padding=1)
            
        # global-context-aware weight prediction network
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        self.labelenc1 = nn.Sequential(norm_layer(nn.Conv2d(self.opt.semantic_nc, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True)) # 256
        self.labelenc2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 128
        self.labelenc3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 64
        self.labelenc4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 32
        self.labelenc5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 16
        self.labelenc6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 8
        if self.num_upsampling_layers == 'more':
            self.labelenc7 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 4

        # lateral for fpn
        self.labellat1 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#16
        self.labellat2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#32
        self.labellat3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#64
        self.labellat4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#128
        self.labellat5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#256
        if self.num_upsampling_layers == 'more':
            self.labellat6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))
        
        self.labeldec1 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        if self.num_upsampling_layers == 'more':
            self.labeldec6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))

        # image generator
        self.head_0 = DepthsepCCBlock(16*nf, 16*nf, opt, opt.semantic_nc + nf)
        self.G_middle_0 = DepthsepCCBlock(16*nf, 16*nf, opt, opt.semantic_nc + nf)
        self.G_middle_1 = DepthsepCCBlock(16*nf, 16*nf, opt, opt.semantic_nc + nf)

        self.up_0 = DepthsepCCBlock(16*nf, 8*nf, opt, opt.semantic_nc + nf)
        self.up_1 = DepthsepCCBlock(8*nf, 4*nf, opt, opt.semantic_nc + nf)
        self.up_2 = DepthsepCCBlock(4*nf, 2*nf, opt, opt.semantic_nc + nf)
        self.up_3 = DepthsepCCBlock(2*nf, 1*nf, opt, opt.semantic_nc + nf)
            
        final_nc = nf
 
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if self.num_upsampling_layers == 'more':
            num_up_layers = 6
        else:
            num_up_layers = 5

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
            
        return sw, sh    

    def forward(self, input, z=None):
        seg = input
        
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16*self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        # encode segmentation labels
        seg1 = self.labelenc1(seg) # 256
        seg2 = self.labelenc2(seg1) # 128
        seg3 = self.labelenc3(seg2) # 64
        seg4 = self.labelenc4(seg3) # 32
        seg5 = self.labelenc5(seg4) # 16
        seg6 = self.labelenc6(seg5) # 8
        if self.num_upsampling_layers == 'more':
            seg7 = self.labelenc7(seg6)
            segout1 = seg7
            segout2 = self.up(segout1) + self.labellat1(seg6) 
            segout2 = self.labeldec1(segout2) 
            segout3 = self.up(segout2) + self.labellat2(seg5) 
            segout3 = self.labeldec2(segout3) 
            segout4 = self.up(segout3) + self.labellat3(seg4) 
            segout4 = self.labeldec3(segout4) 
            segout5 = self.up(segout4) + self.labellat4(seg3) 
            segout5 = self.labeldec4(segout5) 
            segout6 = self.up(segout5) + self.labellat5(seg2) 
            segout6 = self.labeldec5(segout6) 
            segout7 = self.up(segout6) + self.labellat6(seg1) 
            segout7 = self.labeldec6(segout7)
        else:
            segout1 = seg6
            segout2 = self.up(segout1) + self.labellat1(seg5)
            segout2 = self.labeldec1(segout2) 
            segout3 = self.up(segout2) + self.labellat2(seg4) 
            segout3 = self.labeldec2(segout3) 
            segout4 = self.up(segout3) + self.labellat3(seg3)
            segout4 = self.labeldec3(segout4) 
            segout5 = self.up(segout4) + self.labellat4(seg2)
            segout5 = self.labeldec4(segout5) 
            segout6 = self.up(segout5) + self.labellat5(seg1) 
            segout6 = self.labeldec5(segout6) 

        x = self.head_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout1), dim=1)) # 8

        x = self.up(x)
        x = self.G_middle_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout2), dim=1)) # 16
        if self.num_upsampling_layers == 'more':
            x = self.up(x)
            x = self.G_middle_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout3), dim=1)) 
        else:
            x = self.G_middle_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout2), dim=1)) # 16

        x = self.up(x)
        if self.num_upsampling_layers == 'more':
            x = self.up_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout4), dim=1)) # 32
        else:
            x = self.up_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout3), dim=1)) # 32

        x = self.up(x)
        if self.num_upsampling_layers == 'more':
            x = self.up_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout5), dim=1)) # 64
        else:
            x = self.up_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout4), dim=1)) # 64

        x = self.up(x)
        if self.num_upsampling_layers == 'more':
            x = self.up_2(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout6), dim=1)) # 128
        else:
            x = self.up_2(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout5), dim=1)) # 128

        x = self.up(x)
        if self.num_upsampling_layers == 'more':
            x = self.up_3(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout7), dim=1)) # 256
        else:
            x = self.up_3(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout6), dim=1)) # 256


        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x



# feature pyramid light
class DyConvContext4Generator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='more',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16*nf*self.sw*self.sh)            
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16*nf*self.sw*self.sh, 3, padding=1)
            
        # semantic map encoder to incorporate context
        # currently only supports 'normal' mode
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        self.labelenc1 = nn.Sequential(norm_layer(nn.Conv2d(self.opt.semantic_nc, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True)) # 256

        self.labelenc2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 128
        self.labelenc3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 64
        self.labelenc4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 32
        self.labelenc5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 16
        self.labelenc6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 8
        self.labelenc7 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 4

        # lateral for fpn
        self.labellat1 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#16
        self.labellat2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#32
        self.labellat3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#64
        self.labellat4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#128
        self.labellat5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#256
        self.labellat6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))
        # semantic map decoder
        self.labeldec1 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))

        # image generator
        self.head_0 = DepthsepCCBlock(16*nf, 16*nf, opt, opt.semantic_nc + nf)
        self.G_middle_0 = DepthsepCCBlock(16*nf, 16*nf, opt, opt.semantic_nc + nf)
        self.G_middle_1 = DepthsepCCBlock(16*nf, 16*nf, opt, opt.semantic_nc + nf)

        self.up_0 = DepthsepCCBlock(16*nf, 8*nf, opt, opt.semantic_nc + nf)
        self.up_1 = DepthsepCCBlock(8*nf, 4*nf, opt, opt.semantic_nc + nf)
        self.up_2 = DepthsepCCBlock(4*nf, 2*nf, opt, opt.semantic_nc + nf)
        self.up_3 = DepthsepCCBlock(2*nf, 1*nf, opt, opt.semantic_nc + nf)
            
        final_nc = nf
 
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 6

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
            
        return sw, sh    

    def forward(self, input, z=None):
        seg = input
        
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16*self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        # encode segmentation labels
        seg1 = self.labelenc1(seg) # 256
        seg2 = self.labelenc2(seg1) # 128
        seg3 = self.labelenc3(seg2) # 64
        seg4 = self.labelenc4(seg3) # 32
        seg5 = self.labelenc5(seg4) # 16
        seg6 = self.labelenc6(seg5) # 8
        seg7 = self.labelenc7(seg6)
        segout1 = seg7
        segout2 = self.up(segout1) + self.labellat1(seg6) # 8
        segout2 = self.labeldec1(segout2) # 8
        segout3 = self.up(segout2) + self.labellat2(seg5) # 16
        segout3 = self.labeldec2(segout3) # 16
        segout4 = self.up(segout3) + self.labellat3(seg4) # 32
        segout4 = self.labeldec3(segout4) # 32
        segout5 = self.up(segout4) + self.labellat4(seg3) # 64
        segout5 = self.labeldec4(segout5) # 64
        segout6 = self.up(segout5) + self.labellat5(seg2) # 128
        segout6 = self.labeldec5(segout6) # 128
        segout7 = self.up(segout6) + self.labellat6(seg1) # 256
        segout7 = self.labeldec6(segout7)

        x = self.head_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout1), dim=1)) # 8

        x = self.up(x)
        x = self.G_middle_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout2), dim=1)) # 16
        x = self.up(x)

        x = self.G_middle_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout3), dim=1)) # 16

        x = self.up(x)
        x = self.up_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout4), dim=1)) # 32
        x = self.up(x)
        x = self.up_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout5), dim=1)) # 64
        x = self.up(x)
        x = self.up_2(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout6), dim=1)) # 128
        x = self.up(x)
        x = self.up_3(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout7), dim=1)) # 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x

class DyConvContext4cocoGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16*nf*self.sw*self.sh)            
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16*nf, 3, padding=1)
            
        # semantic map encoder to incorporate context
        # currently only supports 'normal' mode
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        self.labelenc1 = nn.Sequential(norm_layer(nn.Conv2d(self.opt.semantic_nc, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True)) # 256
        self.labelenc2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 128
        self.labelenc3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 64
        self.labelenc4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 32
        self.labelenc5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 16
        self.labelenc6 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1, stride=2), opt), nn.LeakyReLU(0.2, True)) # 8

        # lateral for fpn
        self.labellat1 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#16
        self.labellat2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#32
        self.labellat3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#64
        self.labellat4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#128
        self.labellat5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 1), opt), nn.LeakyReLU(0.2, True))#256
        # semantic map decoder
        # currently only supports 'normal' mode
        self.labeldec1 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec2 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec3 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec4 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))
        self.labeldec5 = nn.Sequential(norm_layer(nn.Conv2d(nf, nf, 3, padding=1), opt), nn.LeakyReLU(0.2, True))

        # image generator
        self.head_0 = DepthsepCCBlock(16*nf, 16*nf, opt, opt.semantic_nc + nf)

        self.G_middle_0 = DepthsepCCBlock(16*nf, 16*nf, opt, opt.semantic_nc + nf)
        self.G_middle_1 = DepthsepCCBlock(16*nf, 16*nf, opt, opt.semantic_nc + nf)

        self.up_0 = DepthsepCCBlock(16*nf, 8*nf, opt, opt.semantic_nc + nf)
        self.up_1 = DepthsepCCBlock(8*nf, 4*nf, opt, opt.semantic_nc + nf)
        self.up_2 = DepthsepCCBlock(4*nf, 2*nf, opt, opt.semantic_nc + nf)
        self.up_3 = DepthsepCCBlock(2*nf, 1*nf, opt, opt.semantic_nc + nf)
        
        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
            
        return sw, sh    

    def forward(self, input, z=None):
        seg = input
        
        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16*self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        # encode segmentation labels
        seg1 = self.labelenc1(seg) # 256
        seg2 = self.labelenc2(seg1) # 128
        seg3 = self.labelenc3(seg2) # 64
        seg4 = self.labelenc4(seg3) # 32
        seg5 = self.labelenc5(seg4) # 16
        seg6 = self.labelenc6(seg5) # 8
        segout1 = seg6
        segout2 = self.up(segout1) + self.labellat1(seg5) # 8
        segout2 = self.labeldec1(segout2) # 8
        segout3 = self.up(segout2) + self.labellat2(seg4) # 16
        segout3 = self.labeldec2(segout3) # 16
        segout4 = self.up(segout3) + self.labellat3(seg3) # 32
        segout4 = self.labeldec3(segout4) # 32
        segout5 = self.up(segout4) + self.labellat4(seg2) # 64
        segout5 = self.labeldec4(segout5) # 64
        segout6 = self.up(segout5) + self.labellat5(seg1) # 128
        segout6 = self.labeldec5(segout6) # 128

        x = self.head_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout1), dim=1)) # 8

        x = self.up(x)
        x = self.G_middle_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout2), dim=1)) # 16

        x = self.G_middle_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout2), dim=1)) # 16

        x = self.up(x)
        x = self.up_0(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout3), dim=1)) # 32
        x = self.up(x)
        x = self.up_1(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout4), dim=1)) # 64
        x = self.up(x)
        x = self.up_2(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout5), dim=1)) # 128
        x = self.up(x)
        x = self.up_3(x, torch.cat((F.interpolate(seg, size=x.size()[2:], mode='nearest'), segout6), dim=1)) # 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x

