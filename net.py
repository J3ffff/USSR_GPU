import torch
import torch.nn as nn


class USSRNet_3(nn.Module):
	def __init__(self, input_channels=1, kernel_size=3, channels=64):
		super(USSRNet_3, self).__init__()
		self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.conv7(x)

		return x

class USSRNet_5(nn.Module):
	def __init__(self, input_channels=1, kernel_size=3, channels=64):
		super(USSRNet_5, self).__init__()
		self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

		self.relu = nn.ReLU()
		print(kernel_size)

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.conv7(x)

		return x

class USSRNet_8(nn.Module):
	def __init__(self, input_channels=1, kernel_size=3, channels=64):
		super(USSRNet_8, self).__init__()
		self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

		self.relu = nn.ReLU()
		print(kernel_size)

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		x = self.relu(self.conv5(x))
		x = self.relu(self.conv6(x))
		x = self.conv7(x)

		return x

class USSRNet_10(nn.Module):
	def __init__(self, input_channels=1, kernel_size=3, channels=64):
		super(USSRNet_10, self).__init__()
		self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		self.conv7 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
		self.conv8 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
		self.conv9 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

		self.relu = nn.ReLU()
		print(kernel_size)

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		x = self.relu(self.conv5(x))
		x = self.relu(self.conv6(x))
		x = self.relu(self.conv7(x))
		x = self.relu(self.conv8(x))
		x = self.conv9(x)

		return x

class USSRNet_delated(nn.Module):
	def __init__(self, input_channels=1, kernel_size=3, channels=64, dilation = 2):
		super(USSRNet_delated, self).__init__()
		self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.conv7(x)

		return x

class USSRNet_5_delated(nn.Module):
	def __init__(self, input_channels=1, kernel_size=3, channels=64, dilation = 2):
		super(USSRNet_5_delated, self).__init__()
		self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.conv7(x)

		return x

class USSRNet_8_delated(nn.Module):
	def __init__(self, input_channels=1, kernel_size=3, channels=64, dilation = 2):
		super(USSRNet_8_delated, self).__init__()
		self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		x = self.relu(self.conv5(x))
		x = self.relu(self.conv6(x))
		x = self.conv7(x)

		return x

class USSRNet_10_delated(nn.Module):
	def __init__(self, input_channels=1, kernel_size=3, channels=64, dilation = 2):
		super(USSRNet_10_delated, self).__init__()
		self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv7 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv8 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)
		self.conv9 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=dilation, bias=True, dilation=dilation)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		x = self.relu(self.conv5(x))
		x = self.relu(self.conv6(x))
		x = self.relu(self.conv7(x))
		x = self.relu(self.conv8(x))
		x = self.conv9(x)

		return x