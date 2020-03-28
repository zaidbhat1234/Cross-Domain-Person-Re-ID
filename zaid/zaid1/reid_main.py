from __future__ import print_function, division
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms as trans
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from reid_network import AdaptReID_model
from reid_dataset import AdaptReID_Dataset
from reid_loss import *
import config
from reid_evaluate import evaluate
import test1
import random

parser = ArgumentParser()
parser.add_argument("--use_gpu", default=2, type=int)
parser.add_argument("--source_dataset", choices=['CUHK03', 'Duke', 'Market', 'MSMT17_V1'], type=str)
parser.add_argument("--target_dataset", choices=['CUHK03', 'Duke', 'Market', 'MSMT17_V1'], type=str)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--total_epochs", default=1000, type=int)
parser.add_argument("--w_loss_rec", default=0.1, type=float)
parser.add_argument("--w_loss_dif", default=0.1, type=float)
parser.add_argument("--w_loss_mmd", default=0.1, type=float)
parser.add_argument("--w_loss_ctr", default=1.0, type=float)
parser.add_argument("--dist_metric", choices=['L1', 'L2', 'cosine', 'correlation'], type=str)
parser.add_argument("--rank", default=1, type=int)
parser.add_argument("--model_dir", default='model', type=str)
parser.add_argument("--model_name", default='basic_10cls', type=str)
parser.add_argument("--pretrain_model_name", default=None, type=str)

args = parser.parse_args()
total_batch = 4272
#trans.RandomHorizontalFlip(p=0.5), can be added to transform, was used by ARN
transform_list = [trans.Resize(size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),\
                  trans.ToTensor(), trans.Normalize(mean=config.MEAN, std=config.STD)]



def setup_gpu():
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print('Using GPU: {}'.format(args.use_gpu))
    

def tb_visualize(image):
    """Un-normalises the images"""
    data=[]
    mean1=(0.485, 0.456, 0.406)
    std1=(0.229, 0.224, 0.225)
    for tensor in image:
        coun = -1
        
        data1=[]
        for tensor1 in tensor:
            coun = coun+1
            tensor1 = (tensor1*std1[coun]) + mean1[coun]
            data1.append(tensor1)
        t = torch.stack(data1)
        data.append(t)
    final = torch.stack(data)
            
    image = final
    return image


def get_ind_sal(idxx,step):
    """gets the index of the corresponding images to be retrieved to get saliency images"""
    ind1=[]
    for t in idxx:
        x = t.data[0]
        x = x.numpy()
        ind1.append(x)
    return ind1


def get_batch(data_iter, data_loader):
    """
        Gets batch_size number of images from the corresponding dataset
        """
    try:
        _, batch = next(data_iter)
    except:
        data_iter = enumerate(data_loader)
        _, batch = next(data_iter)
    if batch['image'].size(0) < args.batch_size or batch['image'].size(0) > args.batch_size:
        print("BATCH",batch['image'].size(0))
        batch, data_iter = get_batch(data_iter, data_loader)
    return batch, data_iter

def save_model(model, step):
    model_path = '{}/{}/{}.pth.tar'.format(args.model_dir ,args.model_name ,args.model_name)
    torch.save(model.state_dict(), model_path)

def save_epoch(epoch):
    epoch_path ='{}/{}/{}_{}.pth.tar'.format(args.model_dir ,args.model_name ,args.model_name,epoch)
    torch.save(epoch.state_dict(), epoch_path)


def sal_mul(image1,image2):
    """ Function to multiply the image with its saliency map"""
    image = torch.mul(image1, image2)
    return image



def norm(image):
    """function to normalise the image
        Mean and SD used are stored in mean1 and std1 arrays"""
        
    data=[]
    mean1=(0.485, 0.456, 0.406)
    std1=(0.229, 0.224, 0.225)
    for tensor in image:
        coun = -1
        
        data1=[]
        for tensor1 in tensor:
            coun = coun+1
            tensor1 = (tensor1-mean1[coun]) / std1[coun]
            data1.append(tensor1)
        t = torch.stack(data1)
        data.append(t)
    final = torch.stack(data)

    image = final
    return image


def train():
    print('Model name: {}'.format(args.model_name))
    classifier_output_dim = config.get_dataoutput(args.source_dataset)
    
    model = AdaptReID_model(backbone='resnet-50', classifier_output_dim=classifier_output_dim).cuda()
    # print(model.state_dict())
    
    
    
    """
        to load pretrained model once saved
        """
    if args.pretrain_model_name is not None:
        print("Loading pre-trained model")
        model.load_state_dict(torch.load('{}/{}.pth.tar'.format(args.model_dir, args.pretrain_model_name)))
        
    
    
    sourceData = AdaptReID_Dataset(dataset_name=args.source_dataset, mode='source', transform=trans.Compose(transform_list))
    sourceDataloader = DataLoader(sourceData, batch_size=args.batch_size, shuffle=True)
    source_iter = enumerate(sourceDataloader)
    """
    #DataLoader for target domain dataset
    targetData = AdaptReID_Dataset(dataset_name=args.target_dataset, mode='train',
                               transform=trans.Compose(transform_list))

    targetDataloader = DataLoader(targetData, batch_size=args.batch_size)
    target_iter = enumerate(targetDataloader)"""

    unique_list = sourceData.unique_list
    
    
    #dataloader for saliency maps
    salData = AdaptReID_Dataset(dataset_name=args.source_dataset, mode='sal', transform=trans.Compose(transform_list))
    salDataloader = DataLoader(salData, batch_size=args.batch_size, shuffle=True)
    sal_iter = enumerate(salDataloader)
    
    
    model_opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    writer = SummaryWriter('/media/zaid/zaid1/log/{}'.format(args.model_name))
    match, junk = None, None
    flag1 = 0
    step1 = 1

    while True:
        for batch_idx in range(0,total_batch,step1):
            model.train()

            step =  step1
            
            print('{} step: {}/{} ({:.2f}%)'.format(args.model_name, step+1, total_batch, float(step+1)*100.0/total_batch))

                
            sourceDataloader = DataLoader(sourceData, batch_size=args.batch_size, pin_memory= True)  # Creates iterale dataset , shuffle ensures random selection of batches from dataset
            source_iter = enumerate(sourceDataloader) #total images/batch size = number of time to run in each epoch to test all images.
            source_batch, source_iter = get_batch(source_iter, sourceDataloader)
            source_image, source_label, _,idxx = split_datapack(source_batch)

            
            # to get the saliency maps corresponding to source image
            
            salData.id_sal =get_ind_sal(idxx,step) #store the index of the images taken so as to load same images for corresponding saliency map
            if(len(salData.id_sal)>8): #change 8 to args.batch_size
                salData.id_sal = salData.id_sal[8:]
            sal_batch, sal_iter = get_batch(sal_iter, salDataloader)
            sal_image, sal_label, _,_ = split_datapack(sal_batch)
            if (torch.equal(source_label, sal_label))== False:
                continue
        
            #Multiply image with its corresponding map to remove information outside the map
            img_s = norm( sal_mul(tb_visualize( source_image),tb_visualize( sal_image)))
            
            
            
            #get images for positive anchor
            source_batch_1, source_iter = get_batch(source_iter, sourceDataloader)
            source_image1, source_label1, _,idxx = split_datapack(source_batch_1)
            
            #Saliency maps for positive image
            salData.id_sal = get_ind_sal(idxx,step)
            sal_batch1, sal_iter = get_batch(sal_iter, salDataloader)
            sal_image1, sal_label1, _,_ = split_datapack(sal_batch1)
            if (torch.equal(source_label1, sal_label1))== False:
                continue
            
            img_s1 = norm(sal_mul(tb_visualize(source_image1),tb_visualize(sal_image1)))


            #get images for negative anchor
            source_batch_neg, source_iter = get_batch(source_iter, sourceDataloader)
            source_image_neg, source_label_neg, _,idxx = split_datapack(source_batch_neg)
            #Saliency maps for negative image
            salData.id_sal = get_ind_sal(idxx,step)
            sal_batch2, sal_iter = get_batch(sal_iter, salDataloader)
            sal_image_neg, sal_label_neg, _,_ = split_datapack(sal_batch2)
            if (torch.equal(source_label_neg, sal_label_neg))== False:
                continue

            img_neg =norm(sal_mul(tb_visualize(source_image_neg),tb_visualize(sal_image_neg)))

            
                    
            """
            print("1",source_label,"2",sal_label)
            print("1",source_label1,"2",sal_label1)
            print("11",source_label_neg,"22",sal_label_neg)"""

            step1=step1 +1
            
            

            """target_batch, target_iter = get_batch(target_iter, targetDataloader)
            target_image, target_label, _,idxx = split_datapack(target_batch)"""
            
            #during training only 751 unique id's between 0-1500 are in dataset so alias index has to be obtained for classification loss
            alias_index = source_label
            for ind in range (args.batch_size):
                alias_index[ind] = unique_list.index(source_label[ind].data)
            #alias_index = alias_index%50 #if 50 id's chosen at random
            
            
            
            """
                recon_img = reconstructed image having same pose as positive anchor and appearance as source image
                feature = identity embedding corresponding to source image
                feature1 = identity embedding corresponding to positive anchor
                feature2 = identity embedding corresponding to negative image
                feature_or = pose embedding corresponding to positive anchor
                pred_s = predicted labels for the source image by the classifier in the model
                
                """
            
            recon_img,feature,feature1,feature2,feature_or,pred_s = model(source_img=img_s,target_img = img_s1,negative_img=img_neg, flag = 1)
            
            recon_img1 =sal_mul(recon_img,tb_visualize(sal_image1)) # multiply reconstructed image with the saliency map for masked loss
            
            
            loss_rec = masked_l2_loss(recon_img1,tb_visualize(img_s1),tb_visualize(sal_image1))
            #loss_rec = loss_rec_func(recon_img,tb_visualize(img_s1))
            loss_dif = loss_dif_func(feature,  feature_or) #orthogonality loss
            #loss_trip = loss_triplet1(feature,feature1,feature2) *10 #triplet loss
            loss_cls = loss_cls_func(pred_s, alias_index) #classification loss
            loss_mse_func = torch.nn.MSELoss()
            loss_app = loss_mse_func(feature,feature1) #appearance loss
            loss = loss_cls +loss_rec +loss_dif + loss_app
           
            
            """Writing the images and losses to logs"""
            
            if (step+1)%3000==0:
                writer.add_scalar('loss_ortho', loss_dif, step)
                
                source_image = tb_visualize(img_s) # Un-normalise first before displaying
                source_image_ = make_grid(source_image)
                writer.add_image('source image', source_image_, step)
                """
                    #visualise saliency maps
                sal_image = tb_visualize(sal_image)
                sal_image_ = make_grid(sal_image)
                writer.add_image('saliency image', sal_image_, step)
                
                sal_image1 = tb_visualize(sal_image1)
                sal_image_1 = make_grid(sal_image1)
                writer.add_image('saliency image pos', sal_image_1, step)
                
                sal_image_neg = tb_visualize(sal_image_neg)
                sal_image_neg = make_grid(sal_image_neg)
                writer.add_image('saliency image neg', sal_image_neg, step)
                """
                
                source_image1 = tb_visualize(img_s1)
                source_image_1 = make_grid(source_image1)
                writer.add_image('source image pos', source_image_1, step)

                
                rec_image_ = make_grid(recon_img)
                writer.add_image('rec image', rec_image_, step)
                """
                    #visualise target images
                target_image1 = tb_visualize(target_image)
                target_image_1 = make_grid(target_image1)
                writer.add_image('target domain image', target_image_1, step)"""
                source_image_n = tb_visualize(img_neg)
                source_image_n = make_grid(source_image_n)
                writer.add_image('source image neg', source_image_n, step)
                writer.add_scalar('loss_cls', loss_cls, step)
                writer.add_scalar('loss_rec', loss_rec, step)
                #writer.add_scalar('loss_trip', loss_trip, step)
                writer.add_scalar('loss_app', loss_app, step)
                writer.add_scalar('loss', loss, step)
                save_model(model, step)


            #update model
            model_opt.zero_grad()
            loss.backward()
            model_opt.step()
            
            #evaluate
            if (step+1)%10000==0:
                rank_score, match, junk = evaluate(args, model, transform_list, match, junk)
                writer.add_scalar('rank1_score', rank_score, step)
                print("RANK 5 : ", rank_score)
    #save_model(model, step)


    writer.close()

setup_gpu()
train()
