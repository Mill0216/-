import os
import time
import glob
import json
import random
import itertools
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils import data
from torch.autograd import Variable
from torchvision import transforms


class preprocessed_data(data.Dataset):
    def __init__(self, input_size, r, gray, train, augument, colab):
        self.gray = gray
        self.n_c = 1 if self.gray else 3
        self.input_size = input_size
        self.r = r
        self.augument = augument
        self.colab = colab
        self.train = train
        phase = 'train' if train else 'test'
        self.data_path_list = glob.glob("{}/{}/*".format("dataset" \
            if not colab else "drive/My Drive/Colab Notebooks/dataset", phase))
        
        try:
            self.data_path_list.remove('{}/{}/Icon\r'.format("dataset"\
                if not colab else "drive/My Drive/Colab Notebooks/dataset", phase))
        except:
            pass
        
        random.shuffle(self.data_path_list)
        self.resize = transforms.Compose([transforms.Resize(input_size, Image.NEAREST)])
        
        transform_list = [
            transforms.RandomCrop(tuple([n*r for n in input_size])),
            transforms.RandomVerticalFlip(p=0.5),
        ]
        
        self.len_data_path_list = len(self.data_path_list)
        
        if self.augument:
            transform_list_1 = transform_list
            transform_list_2 = transform_list
            transform_list_1 += [
                transforms.RandomHorizontalFlip(p=1)
            ]
            self.transform1 = transforms.Compose(transform_list_1)
            self.transform2 = transforms.Compose(transform_list_2)
        else:
            transform_list += [
                transforms.RandomHorizontalFlip(p=0.5)
            ]
            self.transform = transforms.Compose(transform_list)
            
        self.crop = transforms.Compose([transforms.RandomCrop(tuple([n*r for n in input_size]))])
        self.toTensor = transforms.Compose([transforms.ToTensor()])
        
        self.i = 0
        
        
    def return_tensor(self, batch_size):
        if self.augument:
            if self.i < self.len_data_path_list:
                transform = self.transform1
            else:
                transform = self.transform2
        elif self.train:
            transform = self.transform
        else:
            transform = self.crop
            
        def np_normalization(np_img, mean, std):
            img_mean = np_img.mean(keepdims=True)
            img_std  = np.std(np_img, keepdims=True)
            return ((np_img-img_mean)/img_std)*std+mean
            
        input_imgs_data = []
        output_imgs_data = []
        datanames = []
        for i in range(batch_size):
            with open(self.data_path_list[self.i+i-1], 'rb') as f:
                img = Image.open(f)
                if not img.format in ['JPEG', 'PNG']:
                    raise NotImplementedError("file '{}' format isn't jpeg or png".format(data_filepath))
                cropped_img_data = img.convert("L") if self.gray else img.convert("RGB")
                cropped_img_data = transform(cropped_img_data)
                for now_input_data,imgs_data in zip([True,False],[input_imgs_data,output_imgs_data]):
                    img_h, img_w = tuple([s if now_input_data else s*self.r for s in self.input_size])
                    img_data = self.resize(cropped_img_data) if now_input_data else cropped_img_data
                    img_data = self.toTensor(img_data).view(self.n_c,img_h,img_w)*2-1
                    imgs_data += [img_data]
                    
            datanames.append(self.data_path_list[self.i+i-1])
        
        return {
            'input' : torch.stack(input_imgs_data,dim=0),
            'output' : torch.stack(output_imgs_data,dim=0),
            'datanames' : datanames
        }

        
    def next(self, batch_size):
        self.i += batch_size
        return self.return_tensor(batch_size)
        
        
    def __len__(self):
        if self.augument:
            return self.len_data_path_list * 2
        else:
            return self.len_data_path_list
        

class ESPCN(nn.Module):
    def __init__(self, input_c, r):
        super(ESPCN, self).__init__()
        
        model = [
            nn.ReflectionPad2d(4),
            nn.Conv2d(input_c, 64, kernel_size=9, stride=1, bias=True),
            nn.Tanh(),
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, bias=True),
            nn.Tanh(),
            nn.ReflectionPad2d(2),
            nn.Conv2d(32, input_c*r*r, kernel_size=5, stride=1, bias=True),
            nn.Tanh(),
            nn.PixelShuffle(r)
        ]
        
        self.model = nn.Sequential(*model)
        
    
    def forward(self, Input):
        return self.model(Input)
    

class ESPCNmodel():
    def __init__(self, gray, r, MODEL_SAVE_DIR):
        self.MODEL_SAVE_DIR = MODEL_SAVE_DIR
        if torch.cuda.is_available():
            print("using gpu")
            self.device = torch.device("cuda")
            print("device:{}".format(torch.cuda.get_device_name(0)))
        else:
            print("using cpu")
            self.device = torch.device("cpu")
            
        def initialize_network(net):
            def initialize_weights(m):
                if hasattr(m, 'weight') and m.__class__.__name__.find('Conv') != -1:
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                    
            if torch.cuda.is_available():
                net = torch.nn.DataParallel(net)
                torch.backends.cudnn.benchmark = True
                net.apply(initialize_weights).cuda(self.device)
            else:
                net.share_memory()
                net.apply(initialize_weights)
            
            return net.type('torch.FloatTensor')
        
        nc = 1 if gray else 3
        
        self.net = initialize_network(ESPCN(nc, r)).to(self.device)
        for param in self.net.parameters():
            param.requires_grad = True
        self.opt = torch.optim.Adam(self.net.parameters(),0.0002,betas=(0.5, 0.999))
        self.lossFunc = nn.MSELoss()
        
    def forward(self):
        self.SRimg = self.net(self.LRimg)
        return self.SRimg
    
    def backward(self, train_now):
        self.loss = self.lossFunc(self.SRimg, self.HRimg)
        if train_now:
            self.opt.zero_grad()
            self.loss.backward()
            self.opt.step()
        return torch.mean(self.loss).item()
        
    
    def train(self):
        self.net.train()
    
    def evaluate(self):
        self.net.eval()
            
    def set_data(self, data):
        self.LRimg = Variable(data['input'].to(self.device), requires_grad=True)
        self.HRimg = Variable(data['output'].to(self.device), requires_grad=True)
        
    def optimize_parameters(self):
        self.forward()
        return self.backward(True)
    
    def evaluate_parameters(self):
        self.forward()
        return self.backward(False)
        
    def update_learning_rate(self, ran_epoch):
        lr = 0.0002 * min([(100-ran_epoch), 50]) * 0.02
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def save_networks(self, ran_epoch):
        save_net_filename = '%s_net.pth' % (ran_epoch)
        save_net_path = os.path.join(self.MODEL_SAVE_DIR, save_net_filename)
        if torch.cuda.is_available():
            torch.save(self.net.cpu().state_dict(), save_net_path)
            self.net.cuda(self.device)
        else:
            torch.save(self.net.state_dict(), save_net_path)
        save_opt_filename = '%s_opt.pth' % (ran_epoch)
        save_opt_path = os.path.join(self.MODEL_SAVE_DIR, save_opt_filename)
        torch.save(self.opt.state_dict(), save_opt_path)
                    
    def load_networks(self, ran_epoch):
        load_net_filename = '%s_net.pth' % (ran_epoch)
        load_net_path = os.path.join(self.MODEL_SAVE_DIR, load_net_filename)
        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module
        state_dict = torch.load(load_net_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.net.load_state_dict(state_dict)
        load_opt_filename = '%s_opt.pth' % (ran_epoch)
        load_opt_path = os.path.join(self.MODEL_SAVE_DIR, load_opt_filename)
        state_dict = torch.load(load_opt_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.opt.load_state_dict(state_dict)
         
    def device_change(self):
        if self.device == torch.device("cuda"):
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            
            
class Progress_Bar():
    def __init__(self, start):
        self.start = start
        self.train_end = False
        
    def __call__(self, i, epoch, datasize, phase):
        per100 = int((i+1)/datasize*100)
        per20 =  int(per100/5)
        if per20 == 20:
            self.taking_time = self.taking_time if self.train_end else time.time() - self.start
            self.train_end = True
            self.start = time.time()
            if phase == 'predicting':
                condition = "all process have been end. time predicting took is %f"%(self.taking_time)
            elif phase == 'evaluating':
                taking_time = time.time() - self.start
                condition = "all process have been end. time training took is {}, evaluating is {}"\
                                .format(self.taking_time, taking_time)
            else:
                condition =  "{} is {}%finished.".format(phase, per100)
        else:
            condition =  "{} is {}%finished.".format(phase, per100)
        print("\r[{0}] epoch{1} {2}".format("＞"*per20 + "＿"*(20-per20),epoch+1,condition),end="")
            
            
class running():
    def __init__(self, r=2, colab=False, gray=False):
        self.r = r
        self.gray = gray
        self.colab = colab
        self.MODEL_SAVE_DIR = "drive/My Drive/Colab Notebooks/model" if colab else "model"
        self.model = ESPCNmodel(gray, r, self.MODEL_SAVE_DIR)
        if gray:
            self.to_pil_img = transforms.Compose([transforms.ToPILImage(mode='L')])
        else:
            self.to_pil_img = transforms.Compose([transforms.ToPILImage(mode='RGB')])
        
        
    def train(self, input_size, ran_epoch=0, batch_size=1, augument=False, trial=True, save_duration=1):
        if save_duration<1 or (type(save_duration) is not int):
            raise NotImplementedError('save_duration must be int and bigger than 0')
        
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        try:
            with open('{}/train_loss_value.json'.format(self.MODEL_SAVE_DIR),'r') as f:
                train_lossvalue_lists = json.load(f)
            with open('{}/valid_loss_value.json'.format(self.MODEL_SAVE_DIR),'r') as f:
                valid_lossvalue_lists = json.load(f)
        except:
            train_lossvalue_lists = {}
            valid_lossvalue_lists = {}
            
        max_epoch = ran_epoch + 2 if trial else 200
        if ran_epoch != 0:
            self.model.load_networks(ran_epoch)
        for epoch in range(ran_epoch, max_epoch):
            self.model.update_learning_rate(epoch)
            
            progress_bar = Progress_Bar(time.time())
            train_dataset = preprocessed_data(input_size,self.r,self.gray,True,augument,self.colab)
            train_losses = []
            train_datasize = 2 if trial else len(train_dataset)
            self.model.train()
            for i in range(train_datasize//batch_size):
                img_data = train_dataset.next(batch_size)
                self.model.set_data(img_data)
                train_loss = self.model.optimize_parameters()
                train_loss_sum = train_loss if i==0 else train_loss_sum + train_loss
                progress_bar(i, epoch, train_datasize//batch_size, 'training')

            train_loss_bar = train_loss_sum / train_datasize

            test_dataset = preprocessed_data(input_size,self.r,self.gray,False,augument,self.colab)
            valid_losses = []
            test_datasize = 2 if trial else len(test_dataset)
            self.model.evaluate()
            for i in range(test_datasize):
                img_data = test_dataset.next(batch_size)
                self.model.set_data(img_data)
                valid_loss = self.model.evaluate_parameters()
                valid_loss_sum = valid_loss if i==0 else valid_loss_sum + valid_loss
                progress_bar(i, epoch, test_datasize, 'evaluating')
            valid_loss_bar = valid_loss_sum / test_datasize

            print()
            ran_epoch += 1
            train_lossvalue_lists['epoch{}'.format(ran_epoch)] = train_loss_bar
            valid_lossvalue_lists['epoch{}'.format(ran_epoch)] = valid_loss_bar
            with open('{}/train_loss_value.json'.format(self.MODEL_SAVE_DIR),'w') as f:
                json.dump(train_lossvalue_lists, f)
            with open('{}/valid_loss_value.json'.format(self.MODEL_SAVE_DIR),'w') as f:
                json.dump(valid_lossvalue_lists, f)
                
            if ran_epoch % save_duration == 0:
                self.model.save_networks(ran_epoch)
            
            
    def predict_save(self, input_size, ran_epoch, train_dir=False, trial=True):
        save_dirname = os.path.join(self.MODEL_SAVE_DIR,
                                    'SR%s(epoch%d'%('train' if train_dir else 'test', ran_epoch))
        os.makedirs(save_dirname, exist_ok=True)
            
        self.model.evaluate()
        self.model.load_networks(ran_epoch)
        
        dataset = preprocessed_data(input_size,self.r,self.gray,train_dir,False,self.colab)
        datasize = 8 if trial else len(dataseta)
        progress_bar = Progress_Bar(time.time())
        for i in range(datasize):
            img_data = dataset.next(1)
            save_filename = img_data['datanames'][0].replace("%s/%s/"%(
                "dataset"if not self.colab else "drive/My Drive/Colab Notebooks/dataset", 
                "train" if train_dir else "test"
            ), '')
            save_filepath = os.path.join(save_dirname, save_filename)
            self.model.set_data(img_data)
            SRimg = self.model.forward()
            SRimg = self.to_pil_img((SRimg.detach().cpu().clone().squeeze()+1)/2)
            SRimg.save(save_filepath, quality= 95)
            progress_bar(i, ran_epoch-1, datasize, 'predicting')
    
    def show_train_curve(self, ran_epoch):
        with open('{}/train_loss_value.json'.format(self.MODEL_SAVE_DIR),'r') as f:
            train_lossvalue_lists = json.load(f)
        with open('{}/valid_loss_value.json'.format(self.MODEL_SAVE_DIR),'r') as f:
            valid_lossvalue_lists = json.load(f)
        
        epoch = len(train_lossvalue_lists.keys())
        x_axis = range(1, ran_epoch+1)
        train_means = []
        valid_means = []
        for j in x_axis:
            train_means += [train_lossvalue_lists['epoch{}'.format(j)]]
            valid_means += [valid_lossvalue_lists['epoch{}'.format(j)]]  
                
        title = 'loss'
        plt.ylim(0, max(train_means+valid_means)*1.1)
        plt.plot(x_axis ,train_means ,label='train')
        plt.plot(x_axis ,valid_means ,label='valid')
        plt.grid()
        plt.legend()
        plt.title(title)
        plt.xlabel('epoch')

