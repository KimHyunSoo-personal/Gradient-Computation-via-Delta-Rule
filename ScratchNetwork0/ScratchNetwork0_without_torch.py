import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def ReLU(x):
    return np.maximum(0, x)

def Sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def Softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

def CNN(xx, weight, bias, padding = 0, index = -1):
    #stride 구현은 못했음
    
    input_shape = xx.shape
    weight_shape = weight.shape
    output_shape = [input_shape[2]+2*padding-weight_shape[2]+1, input_shape[3]+2*padding-weight_shape[3]+1]

    if index == -1:
        x = np.pad(xx, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
    else:
        x_cnn[index] = np.pad(xx, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
    y = np.zeros((input_shape[0], weight_shape[0], output_shape[0], output_shape[1]))

    for batch in range(input_shape[0]):
        for channel in range(weight_shape[0]):
            for i in range(output_shape[0]):
                for j in range(output_shape[1]):
                    if index == -1:
                        y[batch, channel, i, j] = np.sum(x[batch, :, i:i+weight_shape[2], j:j+weight_shape[3]]*weight[channel])+bias[channel]
                    else:
                        y[batch, channel, i, j] = np.sum((x_cnn[index])[batch, :, i:i+weight_shape[2], j:j+weight_shape[3]]*weight[channel])+bias[channel]

    return y

def Convolution_ver1(a, b):
    # a (b, c_before, x, x) : x padding state
    # b (b, c_after, z, z) : delta
    # o (c_after, c_before, o, o) : dW
    # valued convolution

    # dZ/dX * dL/dZ 계산

    a_shape = a.shape
    b_shape = b.shape
    
    output_shape = [a_shape[2]-b_shape[2]+1, a_shape[3]-b_shape[3]+1]

    y = np.zeros((b_shape[1], a_shape[1], output_shape[0], output_shape[1]))

    for channel_a in range(b_shape[1]):
        for channel_b in range(a_shape[1]):
            for i in range(output_shape[0]):
                for j in range(output_shape[1]):
                    y[channel_a, channel_b, i, j] = np.sum(a[:, channel_b, i:i+b_shape[2], j:j+b_shape[3]]*b[:, channel_a, :, :])

    return y

def Convolution_ver2(aa, bb, unpadding=0):
    # a (c_after, c_before, k, k) : W
    # b (b, c_after, z, z) : delta
    # o (b, c_before, o, o) : dX
    # full convolution

    # dZ/dW * dL/dZ 계산

    b_shape = bb.shape
    b = np.flip(bb, axis=(2, 3))
    a_shape = aa.shape
    a = np.pad(aa, ((0, 0), (0, 0), (b_shape[2]-1, b_shape[2]-1), (b_shape[3]-1, b_shape[3]-1)), 'constant', constant_values=0)
    
    output_shape = [a_shape[2]+b_shape[2]-1, a_shape[3]+b_shape[3]-1]

    y = np.zeros((b_shape[0], a_shape[1], output_shape[0], output_shape[1]))

    for batch in range(b_shape[0]):
        for channel in range(a_shape[1]):
            for i in range(output_shape[0]):
                for j in range(output_shape[1]):
                    y[batch, channel, i, j] = np.sum(a[:, channel, i:i+b_shape[2], j:j+b_shape[3]]*b[batch, :, :, :])

    return y[:, :, unpadding:-unpadding, unpadding:-unpadding]

def MaxPool(x, pooling = 1):
    input_shape = x.shape
    y = np.zeros((input_shape[0], input_shape[1], input_shape[2]//pooling, input_shape[3]//pooling))
    output_shape = y.shape
    for batch in range(output_shape[0]):
        for channel in range(output_shape[1]):
            for i in range(output_shape[2]):
                for j in range(output_shape[3]):
                    sample = np.max(x[batch, channel, pooling*i:pooling*(i+1), pooling*j:pooling*(j+1)], axis=0)
                    sample = sample.ravel()
                    y[batch, channel, i, j] = np.max(sample)
    return y

def FullyConnect(xx, weight, bias):
    x = np.matmul(weight, xx.T) + bias[:, np.newaxis]
    x = x.T
    return x


def Forward(x):
    for i in [0, 1, 2]:
        x_relu[i] = CNN(x, conv_weight[i], conv_bias[i], padding=2, index=i)
        x_maxpool[i] = ReLU(x_relu[i])
        x = MaxPool(x_maxpool[i], 2)
    
    x = x.reshape(len(x), -1)

    x_relu[3] = FullyConnect(x, fc1_weight, fc1_bias)
    x = ReLU(x_relu[3])
    x = FullyConnect(x, fc2_weight, fc2_bias)
    x = Softmax(x)

    return x

def Forward_with_image_viewing(xx):
    x = xx[0:1]
    if os.path.exists(f"./Images"):
        shutil.rmtree(f"./Images")
    os.makedirs(f"./Images", exist_ok=True)
    img = np.transpose(x[0], (1, 2, 0))
    img_norm = (img - img.min()) / (img.max() - img.min())
    plt.imsave(f"./Images/original.png", img_norm)

    for i in [0, 1, 2]:
        x_relu[i] = CNN(x, conv_weight[i], conv_bias[i], padding=2, index=i)
        x_maxpool[i] = ReLU(x_relu[i])
        x = MaxPool(x_maxpool[i], 2)
        for j in range(x.shape[1]):
            os.makedirs(f"./Images/CNN{i}", exist_ok=True)
            img = x[0][j]
            if img.max() - img.min() == 0:
                img_norm = np.zeros_like(img)
            else:
                img_norm = (img - img.min()) / (img.max() - img.min())
            plt.imsave(f"./Images/CNN{i}/channel{j}.png", img_norm)
        
    
    x = x.reshape(len(x), -1)

    x_relu[3] = FullyConnect(x, fc1_weight, fc1_bias)
    x = ReLU(x_relu[3])
    x = FullyConnect(x, fc2_weight, fc2_bias)
    x = Softmax(x)

    return x

def Backward(x, y):
    # delta rule 적용하여 각 레이어의 delta를 구한 후 모든 레이어 가중치 gradient 한번에 모아서 연산

    x_cnn[0] = x
    for i in [0, 1, 2]:
        x_relu[i] = CNN(x_cnn[i], conv_weight[i], conv_bias[i], padding=2, index=i)
        x_maxpool[i] = ReLU(x_relu[i])
        x_cnn[i+1] = MaxPool(x_maxpool[i], 2)
    
    x_fc1 = x_cnn[3].reshape(len(x), -1)

    x_relu[3] = FullyConnect(x_fc1, fc1_weight, fc1_bias)
    x_fc2 = ReLU(x_relu[3])
    x = FullyConnect(x_fc2, fc2_weight, fc2_bias)
    d = Softmax(x)

    Loss = -np.sum(y * np.log(d + 1e-9))/x.shape[0]

    #---------

    alpha = 1e-2 #learning rate
    delta2 = (y - d).T #fc2의 delta

    e1 = np.matmul(fc2_weight.T, delta2)
    delta1 = e1 * (x_relu[3] > 0).T #fc1의 delta

    e0 = np.matmul(fc1_weight.T, delta1)

    ce3m = (e0.T).reshape(len(x), 128, 4, 4) #flatten의 delta
    ce3 = np.zeros_like(x_maxpool[2])
    
    window3 = np.zeros_like(x_maxpool[2])
    xmps3 = x_maxpool[2].shape
    for b in range(xmps3[0]):
        for c in range(xmps3[1]):
            for i in range(xmps3[2]//2):
                for j in range(xmps3[3]//2):
                    sample = (x_maxpool[2])[b, c, 2*i:2*(i+1), 2*j:2*(j+1)]
                    mask = (sample == np.max(sample))*1
                    for k in range(2):
                        for l in range(2):
                            window3[b, c, 2*i+k, 2*j+l] = mask[k, l]
                            ce3[b, c, 2*i+k, 2*j+l] = ce3m[b, c, i, j]
    ce3 = ce3*window3 #maxpool_3의 delta
    cdelta3 = ce3*(x_relu[2]>0) #ReLU_3의 delta
    
    #----------

    ce2m = Convolution_ver2(conv_weight[2], cdelta3, unpadding=2) #Conv2d_3의 delta
    ce2 = np.zeros_like(x_maxpool[1])
    
    window2 = np.zeros_like(x_maxpool[1])
    xmps2 = x_maxpool[1].shape
    for b in range(xmps2[0]):
        for c in range(xmps2[1]):
            for i in range(xmps2[2]//2):
                for j in range(xmps2[3]//2):
                    sample = (x_maxpool[1])[b, c, 2*i:2*(i+1), 2*j:2*(j+1)]
                    mask = (sample == np.max(sample))*1
                    for k in range(2):
                        for l in range(2):
                            window2[b, c, 2*i+k, 2*j+l] = mask[k, l]
                            ce2[b, c, 2*i+k, 2*j+l] = ce2m[b, c, i, j]
    ce2 = ce2*window2 #maxpool_2의 delta
    cdelta2 = ce2*(x_relu[1]>0) #ReLU_2의 delta

    #------------

    ce1m = Convolution_ver2(conv_weight[1], cdelta2, unpadding=2) #Conv2d_2의 delta
    ce1 = np.zeros_like(x_maxpool[0])
    
    window1 = np.zeros_like(x_maxpool[0])
    xmps1 = x_maxpool[0].shape
    for b in range(xmps1[0]):
        for c in range(xmps1[1]):
            for i in range(xmps1[2]//2):
                for j in range(xmps1[3]//2):
                    sample = (x_maxpool[0])[b, c, 2*i:2*(i+1), 2*j:2*(j+1)]
                    mask = (sample == np.max(sample))*1
                    for k in range(2):
                        for l in range(2):
                            window1[b, c, 2*i+k, 2*j+l] = mask[k, l]
                            ce1[b, c, 2*i+k, 2*j+l] = ce1m[b, c, i, j]
    ce1 = ce1*window1 #maxpool_1의 delta
    cdelta1 = ce1*(x_relu[0]>0) #ReLU_1의 delta

    #----------
    dW = [
        Convolution_ver1(x_cnn[0], cdelta1) * alpha / x.shape[0],
        Convolution_ver1(x_cnn[1], cdelta2) * alpha / x.shape[0],
        Convolution_ver1(x_cnn[2], cdelta3) * alpha / x.shape[0],
        np.matmul(delta1, x_fc1) * alpha / x.shape[0],
        np.matmul(delta2, x_fc2) * alpha / x.shape[0]
    ]

    db = [
        np.sum(cdelta1, axis=(0, 2, 3)) * alpha / x.shape[0],
        np.sum(cdelta2, axis=(0, 2, 3)) * alpha / x.shape[0],
        np.sum(cdelta3, axis=(0, 2, 3)) * alpha / x.shape[0],
        np.sum(delta1, axis=1) * alpha / x.shape[0],
        np.sum(delta2, axis=1) * alpha / x.shape[0]
    ]

    return dW, db, Loss

if __name__ == "__main__":

    #
    # dataset 준비-------------------------
    #

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    classes = trainset.classes
    print(f"클래스 개수: {len(classes)}개")
    print("샘플 클래스:", classes[:10])

    #
    # Module 준비-------------------------
    #

    channel_list = [32, 64, 128]

    global conv_weight, conv1_bias
    conv_weight = [None] * len(channel_list)
    conv_bias = [None] * len(channel_list)

    for i in range(len(channel_list)):
        if i == 0:
            conv_weight[0] = 0.1*(2*np.random.random((channel_list[i], 3, 5, 5))-1)
        else:
            conv_weight[i] = 0.1*(2*np.random.random((channel_list[i], channel_list[i-1], 5, 5))-1)
        conv_bias[i] = 0.1*(2*np.random.random((channel_list[i]))-1)

    global fc1_weight, fc1_bias
    fc1_weight = 0.1*(2*np.random.random((100, 2048))-1)
    fc1_bias = 0.1*(2*np.random.random((100))-1)

    global fc2_weight, fc2_bias
    fc2_weight = 0.1*(2*np.random.random((10, 100))-1)
    fc2_bias = 0.1*(2*np.random.random((10))-1)

    # x = np.random.random((10, 3, 32, 32))-0.5
    # y = Softmax(np.random.random((10, 10)))

    global x_relu, x_maxpool, x_cnn
    x_relu = [None]*4
    x_maxpool = [None]*3
    x_cnn = [None]*4

    #
    # training -------------------------
    #

    epochs = 10
    for epoch in range(epochs):
        print(f"epoch:{epoch+1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.numpy(), targets.numpy()
            targets_onehot = np.eye(10)[targets]

            #feature image save
            Forward_with_image_viewing(inputs)
            print("feature image updated")

            dW, db, Loss = Backward(inputs, targets_onehot)
            print(f'({batch_idx+1}/{len(trainloader)}) loss:{Loss}')

            fc2_weight += dW[4]
            fc2_bias += db[4]
            fc1_weight += dW[3]
            fc1_bias += db[3]
            
            conv_weight[2] += dW[2]
            conv_bias[2] +=db[2]
            conv_weight[1] += dW[1]
            conv_bias[1] += db[1]
            conv_weight[0] += dW[0]
            conv_bias[0] += db[0]
