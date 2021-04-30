import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# https://github.com/nabsabraham/focal-tversky-unet
# https://github.com/LeeJunHyun/Image_Segmentation

def init_weights(net, init_type='xavier', gain=0.01):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)


class Attention_UNet(nn.Module):
	def __init__(self, in_channels=1, n_classes=2, padding=True,
				 batch_norm=True, up_mode='upconv', residual=False, wf=5, use_attention=False):

		super(Attention_UNet, self).__init__()
		assert up_mode in ('upconv', 'upsample')
		prev_channels = in_channels
		# self.wf = wf
		self.residual = residual
		self.use_attention = use_attention

		self.scaled_input_2_conv = nn.Conv2d(1, 2 ** wf, kernel_size=3, padding=int(padding))
		self.scaled_input_3_conv = nn.Conv2d(1, 2 ** (wf+1), kernel_size=3, padding=int(padding))
		self.scaled_input_4_conv = nn.Conv2d(1, 2 ** (wf+2), kernel_size=3, padding=int(padding))
		self.batch_norm_2 = nn.BatchNorm2d(2 ** (wf))
		self.batch_norm_3 = nn.BatchNorm2d(2 ** (wf+1))
		self.batch_norm_4 = nn.BatchNorm2d(2 ** (wf+2))
	  
		self.Maxpool = nn.MaxPool2d(2, stride=2)
		self.Avgpool = nn.AvgPool2d(2, stride=2)
		self.Relu = nn.ReLU()

		self.conv1 = UNetConvBlock(1, 2 ** (wf), padding, batch_norm, self.residual, first=True)
		self.conv2 = UNetConvBlock(2 ** (wf) + 2 ** (wf), 2 ** (wf+1), padding, batch_norm, self.residual)
		self.conv3 = UNetConvBlock(2 ** (wf+1) + 2 ** (wf+1), 2 ** (wf+2), padding, batch_norm, self.residual)
		self.conv4 = UNetConvBlock(2 ** (wf+2) + 2 ** (wf+2), 2 ** (wf+3), padding, batch_norm, self.residual)

		self.conv5 = UNetConvBlock(2 ** (wf+3), 2 ** (wf+4), padding, batch_norm, self.residual)

		self.Up5 = UNetUpBlock(2 ** (wf+4), 2 ** (wf+3), up_mode, padding, batch_norm)
		self.Att5 = AttnGatingBlock(F_g=2 ** (wf+3),F_l=2 ** (wf+3),F_int=2 ** (wf+2))
		self.Up_conv5 = UNetConvBlock(2 ** (wf+4), 2 ** (wf+3), padding, batch_norm, self.residual)

		self.Up4 = UNetUpBlock(2 ** (wf+3), 2 ** (wf+2), up_mode, padding, batch_norm)
		self.Att4 = AttnGatingBlock(F_g=2 ** (wf+2),F_l=2 ** (wf+2),F_int=2 ** (wf+1))
		self.Up_conv4 = UNetConvBlock(2 ** (wf+3), 2 ** (wf+2), padding, batch_norm, self.residual)

		self.Up3 = UNetUpBlock(2 ** (wf+2), 2 ** (wf+1), up_mode, padding, batch_norm)
		self.Att3 = AttnGatingBlock(F_g=2 ** (wf+1),F_l=2 ** (wf+1),F_int=2 ** (wf))
		self.Up_conv3 = UNetConvBlock(2 ** (wf+2), 2 ** (wf+1), padding, batch_norm, self.residual)

		self.Up2 = UNetUpBlock(2 ** (wf+1), 2 ** (wf), up_mode, padding, batch_norm)
		self.Up_conv2 = UNetConvBlock(2 ** (wf+1), 2 ** (wf), padding, batch_norm, self.residual)

		self.out_conv4 = nn.Conv2d(2 ** (wf+3), n_classes, kernel_size=1)

		self.out_conv3 = nn.Conv2d(2 ** (wf+2), n_classes, kernel_size=1)

		self.out_conv2 = nn.Conv2d(2 ** (wf+1), n_classes, kernel_size=1)

		self.out_conv1 = nn.Conv2d(2 ** (wf), n_classes, kernel_size=1)

		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x):

		blocks = []
		outputs = []

		scaled_input_2 = self.Avgpool(x)
		scaled_input_3 = self.Avgpool(scaled_input_2)
		scaled_input_4 = self.Avgpool(scaled_input_3)

		scaled_input_2 = self.Relu(self.batch_norm_2(self.scaled_input_2_conv(scaled_input_2)))
		scaled_input_3 = self.Relu(self.batch_norm_3(self.scaled_input_3_conv(scaled_input_3)))
		scaled_input_4 = self.Relu(self.batch_norm_4(self.scaled_input_4_conv(scaled_input_4)))

		# encoding path
		x = self.conv1(x)
		blocks.append(x)
		x = self.Maxpool(x)
		x = torch.cat([scaled_input_2, x], 1)

		x = self.conv2(x)
		blocks.append(x)
		x = self.Maxpool(x)
		x = torch.cat([scaled_input_3, x], 1)

		x = self.conv3(x)
		blocks.append(x)
		x = self.Maxpool(x)
		x = torch.cat([scaled_input_4, x], 1)

		x = self.conv4(x)
		blocks.append(x)
		x = self.Maxpool(x)

		x = self.conv5(x)

		 # decoding + concat path
		x = self.Up5(x)
		if self.use_attention is True:
			blocks[-1] = self.Att5(g=x,x=blocks[-1])
		x = torch.cat((blocks[-1],x),dim=1)        
		x = self.Up_conv5(x)
		outputs.append(self.softmax(self.out_conv4(x)))
		
		x = self.Up4(x)
		if self.use_attention is True:
			blocks[-2] = self.Att4(g=x,x=blocks[-2])
		x = torch.cat((blocks[-2],x),dim=1)
		x = self.Up_conv4(x)
		outputs.append(self.softmax(self.out_conv3(x)))
  

		x = self.Up3(x)
		if self.use_attention is True:
			blocks[-3] = self.Att3(g=x,x=blocks[-3])
		x = torch.cat((blocks[-3],x),dim=1)
		x = self.Up_conv3(x)
		outputs.append(self.softmax(self.out_conv2(x)))
	  
	   
		x = self.Up2(x)
		x = torch.cat((blocks[-4],x),dim=1)
		x = self.Up_conv2(x)
		outputs.append(self.softmax(self.out_conv1(x)))
	   
		return outputs


class UNetConvBlock(nn.Module):
	def __init__(self, in_size, out_size, padding, batch_norm, residual=False, first=False):
		super(UNetConvBlock, self).__init__()
		self.residual = residual
		self.out_size = out_size
		self.in_size = in_size
		self.batch_norm = batch_norm
		self.first = first
		self.residual_input_conv = nn.Conv2d(self.in_size, self.out_size, kernel_size=1)
		self.residual_batchnorm = nn.BatchNorm2d(self.out_size)

		if residual:
			padding = 1
		block = []

		if residual and not first:
			block.append(nn.ReLU())
			if batch_norm:
				block.append(nn.BatchNorm2d(in_size))

		block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
							   padding=int(padding)))
		block.append(nn.ReLU())
		if batch_norm:
			block.append(nn.BatchNorm2d(out_size))

		block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
							   padding=int(padding)))

		if not residual:
			block.append(nn.ReLU())
			if batch_norm:
				block.append(nn.BatchNorm2d(out_size))
		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		if self.residual:
			if self.in_size != self.out_size:
				x = self.residual_input_conv(x)
				x = self.residual_batchnorm(x)
			out = out + x

		return out


class UNetUpBlock(nn.Module):
	def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
		super(UNetUpBlock, self).__init__()
		self.in_size = in_size
		self.out_size = out_size

		if up_mode == 'upconv':
			self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
										 stride=2)
		elif up_mode == 'upsample':
			self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
									nn.Conv2d(in_size, out_size, kernel_size=1))

	def forward(self, x):
		out = self.up(x)
		return out

class AttnGatingBlock(nn.Module):
	def __init__(self,F_g,F_l,F_int):
		super(AttnGatingBlock,self).__init__()
		self.W_g = nn.Sequential(
			nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(F_int)
			)
		
		self.W_x = nn.Sequential(
			nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(F_int)
		)

		self.psi = nn.Sequential(
			nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
		)
		
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self,g,x):
		g1 = self.W_g(g)
		x1 = self.W_x(x)
		psi = self.relu(g1+x1)
		psi = self.psi(psi)

		return x*psi

