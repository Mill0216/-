import os
import time
import glob
import json
import random
import itertools
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils import data
from torchvision import transforms


class preprocessed_data(data.Dataset):
    def __init__(self, dataset_name, input_img_size, img_size, 
                 grayscale, train, augumentation, colab, hsv, make_dir=False):
        self.augumentation = augumentation
        self.augumentation_rate = 2 if augumentation else 1
        self.img_size = img_size
        self.grayscale = grayscale
        self.hsv = hsv
        self.make_dir = make_dir
        phase = 'train' if train else 'test'
        self.data_filepath_lists = [[],[]]
        for i, AorB in enumerate(['A','B']):
            self.data_filepath_lists[i] += glob.glob("{}/{}/{}{}/*".format("datasets" \
                if not colab else "drive/My Drive/Colab Notebooks/datasets", 
                                                    dataset_name, phase, AorB))

        for AorB, data_filepath_list in zip(['A', 'B'], self.data_filepath_lists):
            if '{}/{}/{}{}/Icon\r'.format("datasets"\
                if not colab else "drive/My Drive/Colab Notebooks/datasets",
                    dataset_name, phase, AorB) in data_filepath_list:
                
                data_filepath_list.remove('{}/{}/{}{}/Icon\r'.format("datasets"\
                    if not colab else "drive/My Drive/Colab Notebooks/datasets",
                        dataset_name, phase, AorB))
                
        self.dataset_size = max(len(pathlists) for pathlists in self.data_filepath_lists)
        if self.dataset_size < 1:
            raise NotImplementedError("No such a directory like {}"\
                                      .format("{}/{}/{}{}/*".format("datasets" \
                if not colab else "drive/My Drive/Colab Notebooks/datasets", 
                                        dataset_name, phase, AorB)))
        
        for AorB, data_filepath_list in zip(['A', 'B'], self.data_filepath_lists):
            random.shuffle(data_filepath_list)
            if len(data_filepath_list) < self.dataset_size:
                data_filepath_list *= int(self.dataset_size/len(data_filepath_list))+1
            if self.augumentation:
                data_filepath_list *= self.augumentation_rate
            
        self.i = -1         
                    

    def to_tensor(self, i):
        def np_normalization(np_img, mean, std):
            img_mean = np_img.mean(keepdims=True)
            img_std  = np.std(np_img, keepdims=True)
            new_img = ((np_img-img_mean)/img_std)*std+mean
            return np.array([new_img])
        
        def tohsv(np_img):
            nrow = len(np_img)
            nline = len(np_img[0])
            returns = np.empty((nrow, nline, 3), dtype=np.float)
            for i in range(nrow):
                for j in range(nline):
                    r = np.float(np_img[i,j,0])
                    g = np.float(np_img[i,j,1])
                    b = np.float(np_img[i,j,2])
                    MAX = max([r,g,b])
                    MIN = min([r,g,b])
                    if MAX==MIN:
                        H = 0
                    elif MAX==r:
                        H = (g-b)/(MAX-MIN)
                    elif MAX==g:
                        H = (b-r)/(MAX-MIN)+2
                    else:
                        H = (r-g)/(MAX-MIN)+4
                    if H<0:
                        H += 6
                    H = H/3-1
                    if MAX != 0:
                        S = 2*(MAX-MIN)/MAX-1
                    else:
                        S = 0
                    V = 2*MAX/255-1
                    returns[i][j] = np.array([H,S,V])
                    
            return returns
        
        transform_list = [
            transforms.RandomCrop(self.img_size)
        ]
        if not self.make_dir:
            if (not self.augumentation and i>self.dataset_size/2) or\
                            (self.augumentation and i>self.dataset_size):
                transform_list += [transforms.RandomHorizontalFlip(p=1)]
            transform_list += [transforms.RandomVerticalFlip(p=0.5)]
        self.transform = transforms.Compose(transform_list)
        self.totensor = transforms.Compose([transforms.ToTensor()])
        
        tensor_data = {}
        filepath_list = {}

        for AorB, data_filepath_list in zip(['A', 'B'], self.data_filepath_lists):
            with open(data_filepath_list[i], 'rb') as f:
                img = Image.open(f)
                if not img.format in ['JPEG', 'PNG']:
                    raise NotImplementedError("file '{}' format is not JPEG and PNG"\
                                              .format(data_filepath))
                if self.grayscale:
                    np_img = np.array(self.transform(img.convert("L")))
                    tr_img = np_normalization(np_img, 0, 0.5)
                    tensor_img = self.totensor(tr_img).view(1,1,self.img_size,self.img_size)
                else:
                    np_img = np.array(self.transform(img.convert("RGB")))
                    if self.hsv:
                        tr_img = tohsv(np_img)
                    else:
                        tr_img = np_img
                        
                    tensor_img = (self.totensor(tr_img)*2-1)\
                                    .view(1,3,self.img_size,self.img_size)
                    
            tensor_data[AorB] = tensor_img
            filepath_list[AorB] = data_filepath_list[i]

        return {
            'A' : tensor_data['A'], 'B' : tensor_data['B'],
            'A_path' : filepath_list['A'], 'B_path' : filepath_list['B']
        }
    
    def __next__(self):
        self.i += 1
        return self.to_tensor(self.i)
   
    def __len__(self):
        return self.dataset_size*self.augumentation_rate
    
    
class ResnetGenerator(nn.Module):
    def __init__(self, grayscale, img_size, output_nc):
        super(ResnetGenerator, self).__init__()
        input_channel = 1 if grayscale else 3
        n_blocks = 6 if img_size <= 128 else 9

        if not output_nc in [1,3]:
            raise NotImplementedError("output_nc is 1 or 3 (grayscaleoutput or RGBoutput)")

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channel, 64, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        for i in range(2):
            mult = 2**i
            model += [
                nn.Conv2d(64*mult, 64*mult*2, kernel_size=3, 
                          stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(64*mult*2),
                nn.ReLU(True)
            ]
        for i in range(n_blocks):
            model += [
                ResnetBlock()
            ]
        for i in range(2):
            mult = 2**(2-i)
            model += [
                nn.ConvTranspose2d(64*mult, 32*mult, kernel_size=3, stride=2,
                                        padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(32*mult),
                nn.ReLU(True)
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, Input):
        return self.model(Input)
                      
            
class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.model_block = self.build_block()
        
    def build_block(self):
        block = [
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 256, kernel_size=3, padding=0, bias=True),
                nn.InstanceNorm2d(256),
                nn.ReLU(True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 256, kernel_size=3, padding=0, bias=True),
                nn.InstanceNorm2d(256),
        ]
        return nn.Sequential(*block)
    
    def forward(self, x):
        return x + self.model_block(x)
        

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        for i in range(1,3):
            mult = 2**i
            model += [
                nn.Conv2d(32*mult, 64*mult, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(64*mult),
                nn.LeakyReLU(0.2, True)
            ]
        model += [
            nn.Conv2d(64*mult, 64*mult*2, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(64*mult*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64*mult*2, 1, kernel_size=4, stride=1, padding=1),
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, Input):
        return self.model(Input)
    

class HSVLoss(nn.Module):
    def __init__(self):
        super(HSVLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def __call__(self, prediction, real):
        loss_length = len(prediction)
        losses = torch.ones(loss_length)
        for i in range(loss_length):
            HLoss = self.loss(prediction[i][0], real[i][0]) *3
            SLoss = self.loss(prediction[i][1], real[i][1])
            VLoss = self.loss(prediction[i][2], real[i][2])
            losses[i] = (HLoss + SLoss + VLoss)/5
        return losses

    
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.loss = nn.MSELoss()
        
    def __call__(self, prediction, target_is_real, device):
        if target_is_real:
            target_tensor = self.real_label.expand_as(prediction).to(device)
        else:
            target_tensor = self.fake_label.expand_as(prediction).to(device)
        
        loss = self.loss(prediction, target_tensor)
        
        return loss
    
    
class CycleGANmodel():
    def __init__(self, img_size, grayscale, output_nc, identity, hsv):
        """
            netGA : AtoB
            netGB : BtoA
            netDA : discriminator realA and fakeA
            netDB : discriminator realB and fakeB
        """
        if torch.cuda.is_available():
            print("using gpu")
            self.device = torch.device("cuda")
            print("device:{}".format(torch.cuda.get_device_name(0)))
        else:
            print("using cpu")
            self.device = torch.device("cpu")
        self.identity = identity
        self.hsv = hsv

        def initialize_network(net):
            if torch.cuda.is_available():
                net = torch.nn.DataParallel(net)
                torch.backends.cudnn.benchmark = True
            else:
                net.share_memory()
                
            def initialize_weights(m):
                if hasattr(m, 'weight') and m.__class__.__name__.find('Conv') != -1:
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
            
            net.apply(initialize_weights)
            return net
        
        self.netGA = initialize_network(ResnetGenerator(grayscale, img_size, output_nc))
        self.netGB = initialize_network(ResnetGenerator(grayscale, img_size, output_nc))
        # Discriminator's argument, input_nc, is netG's output_nc
        self.netDA = initialize_network(Discriminator(output_nc))
        self.netDB = initialize_network(Discriminator(output_nc))
        self.model_names = ['GA', 'GB', 'DA', 'DB']

        if identity and ((grayscale and output_nc==3) or\
                         (not grayscale and output_nc==1)):
            raise NotImplementedError("taking identityloss, you should set output_nc \
                                                    to the number of input channles")

        self.criterionGANLoss = GANLoss()
        if self.hsv:
            self.criterionCycle = HSVLoss()
            self.criterionIdentity = HSVLoss()
        else:
            self.criterionCycle = nn.L1Loss()
            self.criterionIdentity = nn.L1Loss()
        self.optG = torch.optim.Adam(itertools.chain(self.netGA.parameters(), \
                            self.netGB.parameters()), 0.0002, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(itertools.chain(self.netDA.parameters(), \
                            self.netDB.parameters()), 0.0001, betas=(0.5, 0.999))
        self.optimizers = [self.optG, self.optD]
        self.optimizers_name = ['optG', 'optD']

        self.fake_A_buffer = []
        self.fake_B_buffer = []
            
    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()        
    
    def evaluate(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
            
    def set_input(self, data):
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)

    def run_image_buffer(self, fresh_img, fresh_img_is_fakeA):
        buffer = self.fake_A_buffer if fresh_img_is_fakeA else self.fake_B_buffer
        if len(buffer) < 50:
            buffer.append(fresh_img)
            return fresh_img
        
        else :
            p = random.uniform(0, 1)
            if p > 0.5:
                random_id = random.randint(0, 50 - 1)
                takeout_img = buffer[random_id] 
                #if i==0 else torch.cat([takeout_img, buffer[random_id]])
                buffer[random_id] = fresh_img
                
                return torch.cat([fresh_img, takeout_img], 0)
            else:
                return fresh_img
            
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def update_learning_rate(self, ran_epoch, fine_tuning=False):
        if fine_tuning:
            lr = 0.0002 * max([(100-ran_epoch-1), 1]) * 0.01
        else:
            lr = 0.0002 * max([(200-ran_epoch-1), 1]) * 0.01 if ran_epoch > 100 else 0.0002
        if self.hsv:
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        else:
            for i, optimizer in enumerate(self.optimizers):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr / (i+1)
        
        
    # GAはAをBにする
    # GBはBをAにする
    def forward(self, train):
        self.fake_B = self.netGA(self.real_A)
        self.fake_A = self.netGB(self.real_B)
        if train:
            self.rec_A = self.netGB(self.fake_B)
            self.rec_B = self.netGA(self.fake_A)
        return [self.fake_B, self.fake_A]
        
    def backward_D_basic(self, netD, real, fake, train, ran_epoch, fine_tuning):
        pred_real = netD(real)
        loss_D_real = self.criterionGANLoss(pred_real, True, self.device)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGANLoss(pred_fake, False, self.device)
        loss_D = loss_D_real + loss_D_fake
        self.loss_list += [loss_D]
        if train:
            if self.hsv:
                loss_D.backward()
            else:
                if fine_tuning:
                    if random.uniform(0, 1) > (100-ran_epoch)/100:
                        loss_D *= 0.5
                        loss_D.backward()
                else:
                    if random.uniform(0, 1) > (100-ran_epoch)/100:
                        loss_D *= 0.5
                        loss_D.backward()
        
    def backward_DA(self, ran_epoch, train=True, fine_tuning=False):
        fake_A = self.run_image_buffer(self.fake_A, True)
        self.loss_DA = self.backward_D_basic(self.netDA, self.real_A, 
                                fake_A, train, ran_epoch, fine_tuning)
        
    def backward_DB(self, ran_epoch, train=True, fine_tuning=False):
        fake_B = self.run_image_buffer(self.fake_B, False)
        self.loss_DB = self.backward_D_basic(self.netDB, self.real_B, 
                                fake_B, train, ran_epoch, fine_tuning)
        
    def backward_G(self, train=True):
        if self.identity:
            self.idt_A = self.netGB(self.real_A)
            self.loss_idt_A = self.criterionIdentity(self.idt_A, self.real_A) * 10 * 0.5
            self.idt_B = self.netGA(self.real_B)
            self.loss_idt_B = self.criterionIdentity(self.idt_B, self.real_B) * 10 * 0.5 
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
    
        self.loss_GA = self.criterionGANLoss(self.netDB(self.fake_B), True, self.device)
        self.loss_GB = self.criterionGANLoss(self.netDA(self.fake_A), True, self.device)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * 10
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * 10
        self.loss_G = self.loss_GA + self.loss_GB + self.loss_cycle_A + \
                                self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_list =  [
            self.loss_G.clone(), self.loss_GA.clone(), self.loss_GB.clone(), \
            self.loss_cycle_A.clone(), self.loss_cycle_B.clone(),\
            self.loss_idt_A.clone(), self.loss_idt_B.clone()
        ]
        if train:
            self.loss_G.backward()
        
    def interpolating_predict(self):
        self.fake_A = self.netGA(self.real_A)
        self.fake_B = self.netGA(self.real_B)
        return [self.fake_B, self.fake_A]
    
    def optimize_parameters(self, ran_epoch, fine_tuning):
        self.forward(True)
        self.set_requires_grad([self.netDA, self.netDB], False)
        self.optG.zero_grad()
        self.backward_G()
        self.optG.step()
        self.set_requires_grad([self.netDA, self.netDB], True)
        self.optD.zero_grad()
        self.backward_DA(ran_epoch, True, fine_tuning)
        self.backward_DB(ran_epoch, True, fine_tuning)
        self.optD.step()
        loss_list = [torch.mean(a).item() for a in self.loss_list]
        return loss_list
    
    def evaluate_parameters(self):
        with torch.no_grad():
            self.forward(True)
            self.backward_G(False)
            self.backward_DA('_', False)
            self.backward_DB('_', False)
            loss_list = [torch.mean(a).item() for a in self.loss_list]
        return loss_list
    
    def save_networks(self, MODEL_SAVE_DIR, ran_epoch, ft=False):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (ran_epoch, name) if not ft\
                    else '%s_net_%s_ft.pth'%(ran_epoch, name)
                save_path = os.path.join(MODEL_SAVE_DIR, save_filename)
                net = getattr(self, 'net' + name)
                if torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.device)
                else:
                    torch.save(net.state_dict(), save_path)
        
        for name in self.optimizers_name:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (ran_epoch, name) if not ft\
                    else '%s_net_%s_ft.pth'%(ran_epoch, name)
                save_path = os.path.join(MODEL_SAVE_DIR, save_filename)
                optim = getattr(self, name)
                torch.save(optim.state_dict(), save_path)
         
        save_json = {}
        for AorB, buffer in zip(['A', 'B'], [self.fake_A_buffer, self.fake_B_buffer]):
            save_list = []
            for post_tensor in buffer:
                post_list = post_tensor.cpu().detach().numpy().tolist()
                save_list.append(post_list)
            save_json[AorB] = save_list
        with open('{}/img_buffer.json'.format(MODEL_SAVE_DIR),'w') as f:
            json.dump(save_json, f)

                    
    def load_networks(self, MODEL_SAVE_DIR, ran_epoch=200, ft=False):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (ran_epoch, name) if not ft\
                    else '%s_net_%s_ft.pth'%(ran_epoch, name)
                load_path = os.path.join(MODEL_SAVE_DIR, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module    
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                #for key in state_dict.keys(): 
                #    net.load_state_dict(state_dict)
                net.load_state_dict(state_dict)
                    
        for name in self.optimizers_name:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (ran_epoch, name) if not ft\
                    else '%s_net_%s_ft.pth'%(ran_epoch, name)
                load_path = os.path.join(MODEL_SAVE_DIR, load_filename)
                optim = getattr(self, name)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                #for key in state_dict.keys():
                #    optim.load_state_dict(state_dict)
                optim.load_state_dict(state_dict)
                    
        with open('{}/img_buffer.json'.format(MODEL_SAVE_DIR),'r') as f:
            save_json = json.load(f)
        for key, buffer_list in save_json.items():
            new_buffer = []
            for img_list in buffer_list:
                new_pos_torch = torch.from_numpy(np.array(img_list))
                new_buffer.append(new_pos_torch.float().to(self.device))
            if key == 'A':
                self.fake_A_buffer = new_buffer
            else:
                self.fake_B_buffer = new_buffer
    
    def show_identity(self):
        ide_A = self.netGB(self.real_A)
        ide_B = self.netGA(self.real_B)
        return [ide_B, ide_A]
                    
    def show_cycle(self):
        fakes = self.forward(False)
        cycleA = self.netGB(fakes[0])
        cycleB = self.netGA(fakes[1])
        return [cycleB, cycleA]
        
    def device_change(self):
        if self.device == torch.device("cuda"):
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
                    
                
class running():
    def __init__(self, colab=False, img_size=256, 
                 grayscale=False, output_nc=3, identity=True, hsv=False):
        self.model_arguments = {
            'img_size':img_size,
            'grayscale':grayscale,
            'output_nc':output_nc,
            'identity':identity,
            'hsv':hsv
        }
        
        self.model = CycleGANmodel(**self.model_arguments)
        self.colab = colab
        self.identity = identity
        self.hsv = hsv
        
        self.dataset_arguments = {
            'img_size':img_size,
            'grayscale': grayscale,
            'colab':colab,
            'hsv':hsv
        }
        self.MODEL_SAVE_DIR = "drive/My Drive/Colab Notebooks/model's parameters"\
                                    if colab else "models"
        
    def train(self, origin_dirname=None, ran_epoch=0, input_img_size=256, 
              fine_tuning=False, deriv_ran_epoch=None, deriv_dirname=None, 
                augumentation=False, trial=False, save_duration=1):
        
        if save_duration<1 or type(save_duration) is not int:
            raise NotImplementedError('save_duration must be int and bigger than 0')
        if deriv_dirname == None:
            deriv_dirname = origin_dirname
        if (ran_epoch != 0 and not fine_tuning) or (fine_tuning and deriv_ran_epoch==0):
            self.model.load_networks(os.path.join(
                self.MODEL_SAVE_DIR,origin_dirname), ran_epoch=ran_epoch)
        elif fine_tuning and deriv_ran_epoch != 0:
            self.model.load_networks(os.path.join(
                self.MODEL_SAVE_DIR, deriv_dirname),
                    ran_epoch=deriv_ran_epoch, ft=True)
        MODEL_SAVE_DIR = os.path.join(self.MODEL_SAVE_DIR, deriv_dirname)
        try:
            with open('{}/train_loss_value.json'.format(MODEL_SAVE_DIR),'r') as f:
                train_lossvalue_lists = json.load(f)
            with open('{}/valid_loss_value.json'.format(MODEL_SAVE_DIR),'r') as f:
                valid_lossvalue_lists = json.load(f)
        except:
            train_lossvalue_lists = {}
            valid_lossvalue_lists = {}
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        
        def progress_bar(i, datasize, phase):
            per100 = int((i+1)/datasize*100)
            per20 =  int(per100/5)
            if per20 == 20 and phase == 'evaluating':
                end_of_evaluating = time.time() - evaluating_start
                condition = "all process have been end. time training took is {}, \
                evaluating is {}".format(end_of_training, end_of_evaluating)
            else:
                condition =  "{} is {}%finished.".format(phase, per100)
            print("\r[{0}] epoch{1} {2}"\
                  .format("＞"*per20 + "＿"*(20-per20), epoch+1, condition), end="")

        ran_epoch = deriv_ran_epoch if fine_tuning else ran_epoch
        if trial:
            max_epoch = ran_epoch + 2
        elif fine_tuning:
            max_epoch = 150
        else:
            max_epoch = 250
        for epoch in range(ran_epoch, max_epoch):
            self.model.update_learning_rate(ran_epoch, fine_tuning)
            
            training_start = time.time()
            train_dataset = preprocessed_data(deriv_dirname, input_img_size,
                                        train=True, augumentation=augumentation, 
                                        **self.dataset_arguments)
            train_losses = []
            train_datasize = 2 if trial else len(train_dataset)

            for i in range(train_datasize):
                self.model.train()
                input_data = next(train_dataset)
                self.model.set_input(input_data)
                train_loss = self.model.optimize_parameters(epoch, fine_tuning)
                train_losses = train_loss if i==0 \
                                else [a + b for a, b in zip(train_losses, train_loss)]
                progress_bar(i, train_datasize, 'training')

            train_loss_bar = [loss / train_datasize for loss in train_losses]
            end_of_training = time.time() - training_start


            evaluating_start = time.time()
            test_dataset = preprocessed_data(deriv_dirname, input_img_size, \
                                             train=False, augumentation=False, 
                                             **self.dataset_arguments)
            valid_losses = []
            test_datasize =  2 if trial else len(test_dataset)
            self.model.evaluate()

            for i in range(test_datasize):
                input_data = next(test_dataset)
                self.model.set_input(input_data)
                valid_loss = self.model.evaluate_parameters()
                valid_losses = valid_loss if i==0 \
                    else [a + b for a, b in zip(valid_losses, valid_loss)]
                progress_bar(i, test_datasize, 'evaluating')
            valid_loss_bar = [loss / test_datasize for loss in valid_losses]

            print()
            ran_epoch += 1
            train_lossvalue_lists['epoch{}'.format(ran_epoch)] = train_loss_bar
            valid_lossvalue_lists['epoch{}'.format(ran_epoch)] = valid_loss_bar
            with open('{}/train_loss_value.json'.format(MODEL_SAVE_DIR),'w') as f:
                json.dump(train_lossvalue_lists, f)
            with open('{}/valid_loss_value.json'.format(MODEL_SAVE_DIR),'w') as f:
                json.dump(valid_lossvalue_lists, f)
                
            if ran_epoch % save_duration == 0:
                self.model.save_networks(MODEL_SAVE_DIR, ran_epoch, fine_tuning)
        
            
    def make_directory(self, trial=False, origin_dirname=None, input_img_size=(256,256), 
                        train_data=False, ran_epoch=200, identity='fakes', 
                        deriv_dirname=None, deriv_ran_epoch=None, origin_to_deriv=None):
        #あくまでセーブ用ディレクトリを設定
        if deriv_dirname == None:
            deriv_dirname = origin_dirname
        MODEL_SAVE_DIR = os.path.join(self.MODEL_SAVE_DIR, origin_dirname)
        
        #original_modelとtestdataをダウンロード
        if train_data:
            phase = 'train'
        else:
            phase = 'test'
            
        if deriv_ran_epoch == None or origin_dirname!=deriv_dirname:
            self.model.load_networks(MODEL_SAVE_DIR, ran_epoch)
        else:
            self.model.load_networks(MODEL_SAVE_DIR, deriv_ran_epoch, ft=True)
         
        test_dataset = preprocessed_data(deriv_dirname, input_img_size,
                        train=phase=='train', augumentation=False, make_dir=True,
                            **self.dataset_arguments)
        test_datasize = 8 if trial else len(test_dataset)

        
        def predict_and_imgsave(interpolate, test_data, alpha=0):
            if interpolate:
                model_.set_input(test_data)
                fake_tensors = model_.interpolating_predict()
            else:
                if not interpolate:
                    self.model.set_input(test_data)
                    if identity=='cycle':
                        fake_tensors = self.model.show_cycle()
                    elif identity=='identity':
                        fake_tensors = self.model.show_identity()
                    else:
                        fake_tensors = self.model.forward(False)
                    self.dirheader = identity
                    
                else:
                    model_.set_input(test_data)
                    if identity=='cycle':
                        fake_tensors = model_.show_cycle()
                    elif identity=='identity':
                        fake_tensors = model_.show_identity()
                    else:
                        fake_tensors = model_.forward(False)
                    self.dirheader = identity
                    
            def torgb(np_img):
                nrow = len(np_img)
                nline = len(np_img[0])
                returns = np.empty((nrow, nline, 3), dtype=np.int)
                for i in range(nrow):
                    for j in range(nline):
                        H = (np_img[i,j,0]+1)*180
                        S = (np_img[i,j,1]+1)*127.5
                        V = (np_img[i,j,2]+1)*127.5
                        MAX = V
                        MIN = MAX-((S/255)*MAX)
                        if H<60:
                            r = MAX
                            g = (H/60)*(MAX-MIN)+MIN
                            b = MIN
                        elif H<120:
                            r = ((120-H)/60)*(MAX-MIN)+MIN
                            g = MAX
                            b = MIN
                        elif H<180:
                            r = MIN
                            g = MAX
                            b = ((H-120)/60)*(MAX-MIN)+MIN
                        elif H<240:
                            r = MIN
                            g = ((240-H)/60)*(MAX-MIN)+MIN
                            b = MAX
                        elif H<300:
                            r = ((H-240)/60)*(MAX-MIN)+MIN
                            g = MIN
                            b = MAX
                        else:
                            r = MAX
                            g = MIN
                            b = ((360-H)/60)*(MAX-MIN)+MIN
                        returns[i][j] = np.array([r,g,b])
                return returns
                    
            to_pil_img = transforms.Compose([
                transforms.ToPILImage(mode='RGB')
            ])
            
            if self.hsv:
                fake_imgs = [to_pil_img(torgb(tensor.detach().cpu().clone().numpy()))\
                            for tensor in fake_tensors]
            else:
                fake_imgs = [to_pil_img((tensor.detach().cpu().clone().squeeze()+1)/2)\
                                                                                         for tensor in fake_tensors]

            for AorB, fake_img in zip(['B', 'A'], fake_imgs):
                if interpolate:
                    savedir_name = os.path.join(self.MODEL_SAVE_DIR, deriv_dirname)
                    replaced_name = "datasets/" if not self.colab \
                        else "drive/My Drive/Colab Notebooks/datasets/"+deriv_dirname
                    
                    file_sub = origin_dirname.split('_to_')[0]\
                                        if origin_to_deriv[0]=='A'\
                                            else origin_dirname.split('_to_')[1]
                    file_sub += '2'+deriv_dirname.split('2')[0]\
                                        if origin_to_deriv[0]=='A'\
                                            else deriv_dirname.split('2')[1]
                    filename = test_data['{}_path'.format('A' if AorB=='B' else 'B')]\
                        .replace(replaced_name,'')\
                        .replace("/{}A/".format(phase),'')\
                        .replace("/{}B/".format(phase),'')\
                        .replace(deriv_dirname, '')
                    
                    savedir_name = os.path.join(savedir_name, filename)
                    os.makedirs(savedir_name, exist_ok=True)
                    filename = '{}-alpha={}.jpg'.format(file_sub, alpha)
                else:
                    savedir_name = os.path.join(MODEL_SAVE_DIR, '{}{}(epoch{}'
                                    .format(self.dirheader, AorB+phase, 
                                        ran_epoch if deriv_ran_epoch==None\
                                            else deriv_ran_epoch))
                    os.makedirs(savedir_name, exist_ok=True)
                    replaced_name = "datasets/"+deriv_dirname if not self.colab \
                        else "drive/My Drive/Colab Notebooks/datasets/"+deriv_dirname
                    filename = test_data['{}_path'
                    .format('B' if AorB=='B' and not identity\
                            or AorB=='A' and identity else 'A')]\
                    .replace(replaced_name,'')\
                    .replace("/{}A".format(phase),'')\
                    .replace("/{}B".format(phase),'')\
                    .replace("ide", "cycle")

                fake_img.save(savedir_name+'/'+filename ,quality=95)

        #interpolateならderiv_modelをダウンロードしてinterpolate
        if origin_to_deriv != None:
            interpolate_interval = [round(0.2*r,1) for r in range(5+1)] if trial \
                                        else [round(0.05*r,1) for r in range(20+1)]
            model_ = CycleGANmodel(**self.model_arguments)
            
            test_data = next(test_dataset)
            for alpha in interpolate_interval:
                model_.load_networks(os.path.join(self.MODEL_SAVE_DIR, deriv_dirname),
                                     deriv_ran_epoch, True)

                nets_weights = []
                for i, model in enumerate([self.model, model_]):
                    net = getattr(model, 'netG{}'.format(origin_to_deriv[i]))
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module

                    net_weights = net.cpu().state_dict()
                    if hasattr(net_weights, '_metadata'):
                            del net_weights._metadata
                    nets_weights.append(net_weights)

                net_A = nets_weights[0]
                net_B = nets_weights[1]
                net_interp = OrderedDict()
                for k, v_A in net_A.items():
                    v_B = net_B[k]
                    net_interp[k] = alpha * v_A + (1 - alpha) * v_B

                interpolated_net = getattr(model_, 'netGA')
                if isinstance(interpolated_net, torch.nn.DataParallel):
                        interpolated_net = net.module
                for key in list(net_interp.keys()):
                    interpolated_net.load_state_dict(net_interp)
                    
                predict_and_imgsave(True, test_data, alpha)
                
                
        else:
            self.model.device_change()
            self.model.evaluate()
            for i in range(test_datasize):
                test_data = next(test_dataset)
                predict_and_imgsave(False, test_data)

    
    def show_train_curve(self, dataset_name, ran_epoch):
        MODEL_SAVE_DIR = os.path.join(self.MODEL_SAVE_DIR, dataset_name)
        with open('{}/train_loss_value.json'.format(MODEL_SAVE_DIR),'r') as f:
            train_lossvalue_lists = json.load(f)
        with open('{}/valid_loss_value.json'.format(MODEL_SAVE_DIR),'r') as f:
            valid_lossvalue_lists = json.load(f)
        
        epoch = len(train_lossvalue_lists.keys())
        _, axes = plt.subplots(2,4,figsize=(4 *4, 4*2.5))
        x_axis = range(1, ran_epoch+1)
        for i, ax in enumerate(axes.ravel()):
            train_means = []
            valid_means = []
            if i % 8 == 0:
                # generatorの損失関数の平均
                for j in x_axis:
                    train_means += [train_lossvalue_lists['epoch{}'.format(j)][0]]
                    valid_means += [valid_lossvalue_lists['epoch{}'.format(j)][0]]  
                    
                title = 'G_totalloss'
                ax.set_ylim(0, 10.0)

            elif i % 8 == 1:
                for j in x_axis:
                    DAtrain_mean = [train_lossvalue_lists['epoch{}'.format(j)][7]]
                    DBtrain_mean = [train_lossvalue_lists['epoch{}'.format(j)][8]]
                    DAvalid_mean = [valid_lossvalue_lists['epoch{}'.format(j)][7]]
                    DBvalid_mean = [valid_lossvalue_lists['epoch{}'.format(j)][8]]
                    train_means += [DAtrain_mean[0] + DBtrain_mean[0]]
                    valid_means += [DAvalid_mean[0] + DBvalid_mean[0]]
                title = 'D_totalloss'
                ax.set_ylim(0, 2.0)

            elif i % 8 == 2:
                for j in x_axis:
                    GAtrain_mean = [train_lossvalue_lists['epoch{}'.format(j)][1]]
                    GAvalid_mean = [valid_lossvalue_lists['epoch{}'.format(j)][1]]
                    train_means += [GAtrain_mean]
                    valid_means += [GAvalid_mean]
                title = 'GA_loss'
                ax.set_ylim(0, 1.0)

            elif i % 8 == 3:
                for j in x_axis:
                    GAIdetrain_mean = [train_lossvalue_lists['epoch{}'.format(j)][5]]
                    GAIdevalid_mean = [valid_lossvalue_lists['epoch{}'.format(j)][5]]
                    train_means += [GAIdetrain_mean]
                    valid_means += [GAIdevalid_mean]
                title = 'GAIde_loss'
                ax.set_ylim(0, 2.0)

            elif i % 8 == 4:
                for j in x_axis:
                    ABAtrain_mean = [train_lossvalue_lists['epoch{}'.format(j)][3]]
                    BABtrain_mean = [train_lossvalue_lists['epoch{}'.format(j)][4]]
                    ABAvalid_mean = [valid_lossvalue_lists['epoch{}'.format(j)][3]]
                    BABvalid_mean = [valid_lossvalue_lists['epoch{}'.format(j)][4]]
                    train_means += [ABAtrain_mean[0] + BABtrain_mean[0]]
                    valid_means += [ABAvalid_mean[0] + BABvalid_mean[0]]
                title = 'cycleloss'
                ax.set_ylim(0, 5.0)

            elif i % 8 == 5:
                for j in x_axis:
                    ABtrain_mean = [train_lossvalue_lists['epoch{}'.format(j)][1]]
                    BAtrain_mean = [train_lossvalue_lists['epoch{}'.format(j)][2]]
                    ABvalid_mean = [valid_lossvalue_lists['epoch{}'.format(j)][1]]
                    BAvalid_mean = [valid_lossvalue_lists['epoch{}'.format(j)][2]]
                    train_means += [ABtrain_mean[0] + BAtrain_mean[0]]
                    valid_means += [ABvalid_mean[0] + BAvalid_mean[0]]
                title = 'adversarial_loss'
                ax.set_ylim(0, 2.0)

            elif i % 8 == 6:
                for j in x_axis:
                    GBtrain_mean =  [train_lossvalue_lists['epoch{}'.format(j)][2]]
                    GBvalid_mean =  [valid_lossvalue_lists['epoch{}'.format(j)][2]]
                    train_means += [GBtrain_mean]
                    valid_means += [GBvalid_mean]
                title = 'GB_loss'
                ax.set_ylim(0, 1.0)

            elif i % 8 == 7:
                for j in x_axis:
                    GBIdetrain_mean = [train_lossvalue_lists['epoch{}'.format(j)][6]]
                    GBIdevalid_mean = [valid_lossvalue_lists['epoch{}'.format(j)][6]]
                    train_means += [GBIdetrain_mean]
                    valid_means += [GBIdevalid_mean]
                title = 'GBIde_loss'
                ax.set_ylim(0, 2.0)
                
            ax.plot(x_axis ,train_means ,label='train')
            ax.plot(x_axis ,valid_means ,label='valid')
            ax.grid()
            ax.legend()
            ax.set_title(title)
            ax.set_xlabel('epoch')
            
            
