
import torch
import numpy as np
import torchvision
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

import net#即前面定义网络的python文件（net.py）
#数据预处理

batch_size=64
learning_rate=1e-2
num_epoches=20
#下载训练集MNIST手写数字训练集
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,))  # 拉平
    x = torch.from_numpy(x)
    return x
#data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
train_set=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_set=datasets.MNIST(root='./data',train=False,transform=data_tf)
train_data=DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_data=DataLoader(test_set,batch_size=batch_size,shuffle=False)


#导入网络，定义损失函数和优化方法
model=net.simpleNet(28*28,300,100,10)#简单三层网络
if torch.cuda.is_available():
    model=model.cuda()
criterion=nn.CrossEntropyLoss()#损失函数交叉熵
optimizer=optim.SGD(model.parameters(),lr=learning_rate)#随机梯度下降算法优化损失函数

train_losses=[]
train_acces=[]
eval_losses=[]
eval_acces=[]
for e in range(num_epoches):
    train_loss=0
    train_acc=0
    model.train()
    for data in train_data:
        img,label=data
        #img=img.view(img.size(0),-1)#将每张图片的由二维转为一维
        num=img.numpy().shape[0]
        if torch.cuda.is_available():
            img=Variable(img).cuda()
            label=Variable(label).cuda()
        else:
            im=Variable(img)
            label=Variable(label)
        #前向传播
        out=model(img)
        loss=criterion(out,label)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #记录误差
        train_loss+=loss.data[0]
        #计算分类的准确率
        _,pred=torch.max(out,1)
        num_correct=(pred==label).sum().data[0]
        acc=num_correct/num
        train_acc+=acc
    print(train_loss,train_acc)
    train_losses.append(train_loss/len(train_data))
    train_acces.append(train_acc/len(train_data))

    #在测试集上检验效果
    model.eval()
    eval_loss=0
    eval_acc=0
    for data in test_data:
        img,label=data
        num=img.numpy().shape[0]
        #img=img.view(img.size(0),-1)
        if torch.cuda.is_available():
            img=Variable(img,volatile=True).cuda()#volatile=True表示前向传播时不需要保留缓存，因为对于测试集，不需要做反向传播
            label=Variable(label,volatile=True).cuda()
        else:
            img=Variable(img,volatile=True)
            label=Variable(label,volatile=True)
        out=model(img)
        loss=criterion(out,label)
        # 记录误差
        eval_loss+=loss.data[0]
        #记录准确率
        _,pred=torch.max(out,1)
        num_correct=(pred==label).sum().data[0]
        acc =num_correct/num
        train_acc+=acc
    print (eval_loss,eval_acc)
    eval_losses.append(eval_loss/len(test_data))
    eval_acces.append(eval_acc/len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                     eval_loss / len(test_data), eval_acc / len(test_data)))




















