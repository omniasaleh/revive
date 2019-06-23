import numpy as np
import torch
from math import log10
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
## Options
import argparse
## Utils
from skimage import io, transform, color
import cv2
from matplotlib import pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
## Loading Data
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time

parser = {
  'loadSize': 224,
  'fineSize': 176,
  'seed': 123,
  'batchSize': 70,
  'testBatchSize': 35,
  'nEpochs': 14,
  'threads': 4,
  'lr': 0.0001,
  'pro':1.0,
  'wd':10e-3

}
opt = parser
def rgb2lab(rgb_imgs):
    # rgb_imgs shape: (N, H, W, C)
    # returned images shape: (N, C, H, W)
    lab_imgs = np.zeros(rgb_imgs.shape, dtype=np.float)

    i = 0
    # print("size",lab_imgs.size)
    for rgb_img in rgb_imgs:
        lab_imgs[i, :, :, :] = color.rgb2lab(rgb_img)
        i += 1
    
    # Change the shape of images from (N, H, W, C) to (N, C, H, W)
    return np.rollaxis(lab_imgs, 3, 1)

def lab2rgb(imgs):
    # rgb_imgs shape: (N, C, H, W)
    # returned images shape: (N, C, H, W)

    imgs = np.rollaxis(imgs, 1, 4)
    rgb_imgs = np.zeros(imgs.shape)
    i = 0
    for lab_img in imgs:
        rgb_imgs[i, :, :, :] = color.lab2rgb(lab_img)
        i += 1
    rgb_imgs = rgb_imgs*255
    # Change the shape of images from (N, H, W, C) to (N, C, H, W)
    return np.rollaxis(rgb_imgs, 3, 1)

#
def rgb2gray(rgb_imgs):
    N, H, W, C = rgb_imgs.shape
    gray_imgs = np.zeros((N, 1, H, W))
    i = 0
    for rgb_img in rgb_imgs:
        gray_imgs[i] = color.rgb2gray(rgb_img)
        i += 1
    return gray_imgs


def put_hints(AB, BW):
    N, C, H, W = AB.shape
    # print(BW.shape)
    hints_imgs = np.zeros(AB.shape)
    mask_imgs = np.zeros(BW.shape)

    for i in range(N):
        points = np.random.geometric(opt['pro'])
        # print("Points: ", points)

        for z in range(points-1):
            patch_size = np.random.choice([8,9])

            h = int(np.clip(np.random.normal((H - patch_size + 1) / 2., (H - patch_size + 1) / 4.), 0, H - patch_size))
            w = int(np.clip(np.random.normal((H - patch_size + 1) / 2., (H - patch_size + 1) / 4.), 0, H - patch_size))
            # print("Point Size: ", patch_size, " H: ", h, " W: ", w)
            hints_imgs[i, :, h:h + patch_size, w:w + patch_size] = AB[i, :, h:h + patch_size, w:w + patch_size]

            mask_imgs[i, :, h:h + patch_size, w:w + patch_size] = 1.
    mask_imgs-=.5
    return hints_imgs, mask_imgs


def convert_to_numpy(ic):
    # converts images from ImageCollection type to numpy array type
    return ic.concatenate()

def imshow_img(img):
    img = np.rollaxis(img, 0, 3)
    plt.imshow(img)
    plt.show()

def imshow(bw_img, mask_img, out_img, in_img):
    out_img = np.rollaxis(out_img, 0, 3)
    in_img = np.rollaxis(in_img, 0, 3)
    # print(img.shape)
    fig = plt.figure()
    
    a = fig.add_subplot(1, 4, 1)
    imgplot = plt.imshow(bw_img)
    a.set_title('Gray')
    
    a = fig.add_subplot(1, 4, 2)
    imgplot = plt.imshow(mask_img)
    a.set_title('Mask')
    
    a = fig.add_subplot(1, 4, 3)
    imgplot = plt.imshow(out_img)
    a.set_title('Ground Truth')
    
    a = fig.add_subplot(1, 4, 4)
    imgplot = plt.imshow(in_img)
    a.set_title('Colorized')
    plt.show()


def preprocessing(data):
    data = data.permute(0, 2, 3, 1)#shape(N,H,W,3)
    lab_images = rgb2lab(data.cpu())  # shape (N, 3, H, W)
    gray_images = (lab_images[:, 0:1, :, :]-50.0)/100.0
    ab_images = lab_images[:, 1:, :, :]/110.  # shape (N, 2, H, W)
    hints_images, mask_images = put_hints(ab_images, gray_images)
    result = torch.cat((torch.from_numpy(gray_images), torch.from_numpy(hints_images), torch.from_numpy(mask_images)), dim=1).float()
    
    ground_truth = torch.from_numpy(ab_images).float()
    
    return (result, ground_truth)  # shape (N, 4, H, W)

def get_samplers(dataset_size):
    indices = list(range(dataset_size))
    validation_split = 0.0
    shuffle_dataset = True
    random_seed = 42
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


def load_data(opt):
    datapath =  'data/ResizedDataset_256_256'
    print("ff")

    dataset = torchvision.datasets.ImageFolder(datapath,
                                               transform=transforms.Compose([
                                                   transforms.RandomChoice(
                                                       [transforms.Resize(opt['loadSize'], interpolation=1),
                                                        transforms.Resize(opt['loadSize'], interpolation=2),
                                                        transforms.Resize(opt['loadSize'], interpolation=3),
                                                        transforms.Resize((opt['loadSize'], opt['loadSize']),
                                                                          interpolation=1),
                                                        transforms.Resize((opt['loadSize'], opt['loadSize']),
                                                                          interpolation=2),
                                                        transforms.Resize((opt['loadSize'], opt['loadSize']),
                                                                          interpolation=3)]),
                                                   transforms.RandomChoice(
                                                       [transforms.RandomResizedCrop(opt['fineSize'], interpolation=1),
                                                        transforms.RandomResizedCrop(opt['fineSize'], interpolation=2),
                                                        transforms.RandomResizedCrop(opt['fineSize'], interpolation=3)]),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor()]))
    

    return dataset
import numpy as np
import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self,classification=True):
        super(Unet, self).__init__()
        self.classification=classification
        use_bias = True
        self.delta = 0.01
        #   Convolution Block #1
        self.model1=nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1,stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.BatchNorm2d(64,affine=True)

        )
        self.model2=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.BatchNorm2d(128,affine=True)

        )
        self.model3=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.BatchNorm2d(256,affine=True)

        )
        self.model4=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.BatchNorm2d(512,affine=True)

        )
        self.model5=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias),
            nn.ReLU(True),
            nn.BatchNorm2d(512,affine=True)

        )
        self.model6=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=use_bias),
            nn.ReLU(True),
            nn.BatchNorm2d(512,affine=True)

        )
        self.model7=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            nn.BatchNorm2d(512, affine=True)
           
        )
        self.model8=nn.Sequential(
             nn.ReLU(True),
             nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias),
             nn.ReLU(True),
             nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias),
             nn.ReLU(True),
             nn.BatchNorm2d(256, affine=True)
        )
        self.model9=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.ReLU(True),
            
            nn.BatchNorm2d(128, affine=True)
        )
        self.model10=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=use_bias),
            nn.LeakyReLU(negative_slope=.2),
            nn.BatchNorm2d(128, affine=True)
        )
        self.down1=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1,stride=2,bias=False,padding=0)
        )
        #   Convolution Block #2
      
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=2, bias=False, padding=0)
        )
        #   Convolution Block #3
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=2, bias=False, padding=0)
        )
       
        #   Short Cut Between Block #3 and Block #8
        self.conv3short8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=use_bias)
        )

        #   Convolution Block #8
        self.up8_1=nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2, bias=use_bias)
        )
        #   Short Cut Between Block #2 and Block #9
        self.conv2short9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=use_bias)
        )

        #   Convolution Block #9
        self.up9_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2, bias=use_bias)
        )
      
        #   Short Cut Between Block #1 and Block #10
        self.conv1short10 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=use_bias)
        )

        #   Convolution Block #10
        self.up10_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=use_bias)
        )     
        #   Output
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=3, padding=1, stride=1, bias=use_bias)
        )
        # model class
        self.model_class=nn.Sequential(
          nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias)
        )

    def forward(self, x):
      out1= self.model1(x)

      # downsampling between block #1, #2
      down_x = self.down1(out1)
      #print(down_x.shape)

      out2= self.model2(down_x)
      # downsampling between block #2, #3

      down_x = self.down2(out2)
      #print(down_x.shape)

      out3 = self.model3(down_x)
      
      # downsampling between block #3, #4
      down_x = self.down3(out3)
      #print(down_x.shape)

      out4=self.model4(down_x)

      out5=self.model5(out4)
      out6=self.model6(out5)
      out7=self.model7(out6)
      #print(x.shape, x_3.shape)
      # upsampling between block #7, #8
      x = self.up8_1(out7) + self.conv3short8(out3)
      out8 = self.model8(x)
      
      if(self.classification):
            out_class = self.model_class(out8)
            conv9_up = self.up9_1(out8.detach()) + self.conv2short9(out2.detach())
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.up10_1(conv9_3) + self.conv1short10(out1.detach())
            conv10_2 = self.model10(conv10_up)
            out_reg = torch.tanh(self.conv_out(conv10_2))
      else:
            out_class = self.model_class(out8.detach())

            conv9_up = self.model9up(out8) + self.conv2short9(out2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.conv1short10(out1)
            conv10_2 = self.model10(conv10_up)
            out_reg= torch.tanh(self.conv_out(conv10_2))

      return (out_class, out_reg)
     

      
    

    
from torch.nn import init

#train_losses,test_losses=[],[]
torch.manual_seed(opt['seed'])

device = torch.device("cpu")

# loader = torch.utils.data.TensorDataset(ic)
data = load_data(opt)

dataset_size = len(data)

print(dataset_size)

train_sampler, valid_sampler = get_samplers(dataset_size)

print('===> Loading datasets')
training_data_loader = DataLoader(dataset=data,
                                  batch_size=opt['batchSize'], 
                                  sampler=train_sampler,num_workers=9)
#testing_data_loader = DataLoader(dataset=data, batch_size=opt['testBatchSize'], sampler=valid_sampler)

print('===> Building model')
model = Unet()
if type(model) == nn.Conv2d:
  init.normal_(model.weight.data, 0.0, .02)
if type(model) == nn.BatchNorm2d:
  init.normal_(model.weight.data, 0.0, .02)
optimizer = optim.Adam(model.parameters(), lr=opt['lr'],betas=(.9,.999))

state_dict = torch.load('intr_model_epoch_9.pth')
model.load_state_dict(state_dict['state_dict'])
optimizer.load_state_dict(state_dict['optimizer_state_dict'])
model.train()

def encode_ab_ind(data_ab):
    # Encode ab value into an index
    # INPUTS
    #   data_ab   Nx2xHxW \in [-1,1]
    # OUTPUTS
    #   data_q    Nx1xHxW \in [0,Q)

    data_ab_rs = torch.round((data_ab*110.0 + 110.0)/10.0) # normalized bin number
    data_q = data_ab_rs[:,[0],:,:]*23 + data_ab_rs[:,[1],:,:]
    return data_q
  

#criterion = nn.SmoothL1Loss()
criterion = nn.L1Loss()
#criterion_test = nn.MSELoss()
criterionCE=nn.CrossEntropyLoss()
loss_train=0

def train(epoch):
    loss_train = 0
    sec_batch = time.time()
    secs = time.time()
    for iteration, data in enumerate(training_data_loader, 1):
        seconds_beg = time.time()
        print ("Batch Data Load Time", seconds_beg - secs)
        
        inputs, ground_truth = preprocessing(data[0])
        real_B_enc= encode_ab_ind(ground_truth[:, :, ::4, ::4])
        seconds_preprocessing = time.time()
        print ("Preprocessing Time:", seconds_preprocessing - seconds_beg)
        
        optimizer.zero_grad()
        fake_B_class, output = model(inputs)
        seconds_disc = time.time()
        print ("model Time:", seconds_disc - seconds_preprocessing)
#         if iteration%200==0:
#            checkpoint(epoch,loss_train,0,0)
        output.retain_grad()
        loss_G_CE = criterionCE(fake_B_class.type(torch.FloatTensor),
                                          real_B_enc[:, 0, :, :].type(torch.LongTensor))          
        loss = 10 * torch.mean(criterion(output, ground_truth))+loss_G_CE 
        loss.retain_grad()
        loss_train += loss.item()
        loss.backward()
        optimizer.step()
        seconds_loss = time.time()
        print ("loss Time:", seconds_loss-seconds_disc)
        print ("Batch Time:", time.time() - seconds_beg)
        print ("---------------------------------------")
        secs = time.time()
        
   
    loss_train/= len(training_data_loader)
   # train_losses.append(epoch_loss)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, loss_train))
    
# avg_psnr = 0
# loss_test=0

# def test():
#     loss_test=0
#     avg_psnr = 0
#     with torch.no_grad():
#         i = 0
#         for data in testing_data_loader:
#             data2 = data[0]

#             inputs, ground_truth = preprocessing(data2)

            
#             outputs = model(inputs)

#             ground_truth *= 110
#             loss= criterion(outputs, ground_truth)
#             loss_test+=loss.item()
#             mse = criterion_test(outputs, ground_truth)
#             psnr = 10 * log10(110 * 110 / mse.item())
#             avg_psnr += psnr
#             if i%10 == 0:
#                 inputs = inputs.cpu().numpy()
#                 new_out = np.zeros((inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]))
#                 new_out[:, 0] = inputs[:, 0] * 100

#                 new_out = torch.from_numpy(new_out).cpu()

#                 outme = torch.cat((new_out.double(), outputs.cpu().double()), dim=1).cpu().numpy()
#                 outputs = lab2rgb(outme)
#                 outputs = outputs.astype(int)
#                 truth = data2.cpu().numpy()
#                 imshow(inputs[0][0]*100, inputs[0][3]*100, truth[0], outputs[0])
    #        i += 1
#     loss_test/= len(testing_data_loader)
#     #test_losses.append(loss_test)
#     print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch,loss,loss_test,psnr):
    model_out_path = "intractive/intr_model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
   
    torch.save({'state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss,'test_loss':loss_test,'pnsr':psnr}, model_out_path)
    
    print("Checkpoint saved to {}".format(model_out_path))
    
def plot_results(train_losses,test_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


for epoch in range(5, opt['nEpochs'] + 1):
    print('Epoch:', epoch)
    train(epoch)
    #test()
    checkpoint(epoch,loss_train,0,0)
    #plot_results(train_losses,test_losses)
    
    
    
