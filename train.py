
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import time
sys.path.append("..")
#from tools.generate_txt import gen_txt_label_muti_file
#from tools.generate_txt import gen_txt_label
from tools.mydatasets_label import MyDataset

#from _train_model.tools.generate_txt import gen_txt,gen_txt_label
#from _train_model.tools.mydatasets_label import MyDataset
#from train_model.tools import myresnet
# 0.编辑配置信息，
# gpu标志位
gpu_available = torch.cuda.is_available()
# 网络模型列表

#     图片存储文件夹，
#image_dir = "../image/250_bremen"

# image_dir_list="../image/optimal_encode_4multi_dataset/coco_256"
image_path_txt = "../txt/500_375_512_label_image_path_list.txt"
# image_path_txt = "../txt/coco.txt"
# 预训练模型路径
# pthfile = "/home/lemon/Documents/project/model_zoo/ResNet/resnet18-5c106cde.pth"
save_model_name ="../model/Maltese_dog_resnet18_class512_"


batch_size = 8
# 1.生成图像路径信息的txt
#gen_txt_label(image_path_txt, image_dir)
#gen_txt_label_muti_file(image_path_txt,image_dir_list)
# 2.制作dataloader
train_transformer = transforms.Compose([
    # transforms.Resize([250, 250]),
    transforms.TenCrop([ 350,490]),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])) # returns a 4D tensor
    # transforms.ToTensor(),
])
target_transformer = transforms.Compose([
    transforms.Normalize,
    transforms.ToTensor
])
train_dataset = MyDataset(txt_path=image_path_txt, transform=train_transformer,target_transform=target_transformer)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# 3.实例化网络对象
net_name = "resnet18"
net_model = None

net_model = models.resnet18(num_classes=512).cuda()
# net_model.load_state_dict(torch.load("../model/Maltese_dog_92_error=0.pkl"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net_model.parameters(), lr=0.01)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

start = time.clock()
#loss = 0
for epoch in range(2000):
    error_num = 0
    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率
    loss_sum = 0
    correct_num = 0
    # train
    # 4.将dataloader送入网络，查看结果
    for batch_id, (input, label) in enumerate(train_dataloader):

        if gpu_available:
            input = Variable(input.cuda())
            label = Variable(label.cuda())
        else:
            input = Variable(input)
            label = Variable(label)




            #label = label.to(dtype=torch.int64)
        optimizer.zero_grad()

        bs, ncrops, c, h, w = input.size()

        output = net_model(input.view(-1, c, h, w))
        output_avg = output.view(bs, ncrops, -1).mean(1)  # avg over crops
        # output_avg = output.view(bs,-1)  # avg over crops
        # output_avg = output.view(10,-1).mean(1)
        loss = criterion(output_avg, label) #
        loss_sum += loss
        loss.backward()
        optimizer.step()
        pred = output_avg.max(1,keepdim=True)[1]
        pred = pred.view(-1)

        # if int(label) != int(pred):
        #     error_num += 1

        # correct_num +=  torch.eq(pred,label).sum().item()
        # print(pred.shape,label.shape)
        # print(batch_id*batch_size)
        # print(pred,label)
        eq_mat = torch.eq(pred, label)
        # print(eq_mat,len(pred),len(label))
        sum = eq_mat.sum()
        # print(sum)
        sum = sum.item()
        error_num += len(pred) - sum

        if batch_id%10 == 0:

            print("Train Epoch:{}\tbatch_id:{} \tLoss:{:.6f}\terror_num:{}\tpred:{}\tlabel:{}".format(
                epoch,
                batch_id,
                loss.item(),
                error_num,
                pred[1:5],
                label[1:5]
            ))
            # torch.save(net_model.state_dict(), save_model_name)
    if epoch%81 == 0 :
        torch.save(net_model.state_dict(), save_model_name+str(epoch)+".pkl")
    loss_avg = loss_sum/len(train_dataloader.dataset)
    print("epoch:"+str(epoch)+"\tloss_avg:"+str(float(loss_avg))+"\terror_num："+str(error_num))

    if error_num == 0:
    #if epoch == 0:
        torch.save(net_model.state_dict(),save_model_name+str(epoch)+"_error=0.pkl")
        end = time.clock()
        print("epoch:{}, time:{}".format(epoch,end-start))
        break
     

