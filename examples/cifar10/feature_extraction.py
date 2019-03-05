# -*- coding=utf-8 -*-

import sys
caffe_root= '/home/fantasy/caffe/'
import caffe
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import cv2

plt.rcParams['figure.figsize'] = (8,8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_gpu()
model_def = os.path.join(caffe_root,"examples/cifar10/cifar10_quick.prototxt")
model_weights = os.path.join(caffe_root, "examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5")
net = caffe.Net(model_def, model_weights, caffe.TEST)

# 显示各个层的数据类型
print("output:[batch-sizes, channels, height, width]")
for layer_name, blob in net.blobs.items():
    print("output:\t", layer_name, "\t", blob.data.shape)
# 显示各层的参数层及其类型：weight
print("weight:[out-channels, in-channels, filter-height, fiter-width]")
for name, data in net.params.items():
    print("weigth:\t", name, "\t", data[0].data.shape, "\tdiff:\t", data[0].diff.shape)
print("bias:[out-channels]")
for name, data in net.params.items():
    print("bias\t", name, "\t", data[1].data.shape, "\tdiff\t", data[1].diff.shape)


#GraphViz + caffe 的draw_net.py 可以形成可视化的网络模型
#--rankdir 的参数表示网络方向，BT代表图片上的磨练从底到顶绘出
#python python/draw_net.py examples/cifar10/cifar10_quick_solver.prototxt examples/cifar10/cifar-quick.png --rankdir-BT

#显示图片
#img_path = os.path.join(caffe_root, "examples/cifar10/cifar-quick.png")
#net_im = mpimg.imread(img_path)
#plt.imshow(net_im)
#plg.axis('off')

'''
# 待测试的图片
'''
#test_img = os.path.join(caffe_root, "examples/images/cat.jpg")
#test_img = os.path.join(caffe_root, "examples/images/bird.jpg")
#test_img = os.path.join(caffe_root, "examples/images/plane.jpg")
#test_img = os.path.join(caffe_root, "examples/images/cat1.jpg")
test_img = os.path.join(caffe_root, "examples/images/dog.jpg")
#test_img = os.path.join(caffe_root, "examples/images/horse.jpg")
im = caffe.io.load_image(test_img)  #加载图片 
# #print(im.shape)
# plt.imshow(im)
# plt.axis('off')
# plt.show()

def convert_mean(binMean, npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_Mean = open(binMean, 'rb').read()
    blob.ParseFromString(bin_Mean)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    npy_mean=arr[0]
    np.save(npyMean, npy_mean)

binMean = os.path.join(caffe_root, "examples/cifar10/mean.binaryproto")
npyMean = os.path.join(caffe_root, "examples/cifar10/mean.npy")
convert_mean(binMean, npyMean) 

'''
#图片预处理设置 
'''
# 网络的输入图片尺寸 (1,3,32,32)
#print("shape:", net.blobs['data'].data.shape)

#设定图片的shape格式(1,3,32,32) 
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape}) 

#改变维度的顺序，由原始图片(32,32,3)变为(3,32,32)   
transformer.set_transpose('data',(2,0,1)) 

#减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_mean('data', np.load(npyMean).mean(1).mean(1))  

# 缩放到【0，255】之间 
transformer.set_raw_scale('data', 255)  

#交换通道，将图片由RGB变为BGR 
transformer.set_channel_swap('data',(2,1,0))  
#print("source:", im)  # 原始图片数据

#执行上面设置的图片预处理操作，并将图片载入到blob中 
net.blobs['data'].data[...] = transformer.preprocess('data',im)  
inputData = net.blobs['data'].data
#print("preprocess", inputData[0]) # 输入到网络中的图片数据

# plt.figure()
# plt.subplot(1,2,1)
# plt.title("orgin")
# plt.imshow(im)
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.title('subtract mean')
# im2 = transformer.deprocess('data', inputData[0])
# print("deprocess", im2)
# plt.imshow(im2)
# plt.axis('off')
# plt.show()
# print("shape for org:preprocess", im.shape, im2.shape)
# # im为原图, im2为预处理后的图(32,32,3)

def show_data(name, data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0,n**2-data.shape[0]), (0, padsize), (0, padsize)) + ((0,0),)*(data.ndim-3)
    data = np.pad(data, padding, mode="constant", constant_values=(padval, padval))
    
    data = data.reshape((n,n) + data.shape[1:]).transpose((0,2,1,3) + tuple(range(4, data.ndim+1)))
    data = data.reshape((n*data.shape[1], n*data.shape[3])+data.shape[4:])
    plt.figure()
    plt.title(name)
    plt.imshow(data, cmap="gray")
    plt.axis('off')
    plt.show()

# 执行测试
net.forward()

# # 显示conv1的output与weight
# show_data("conv1 output", net.blobs['conv1'].data[0])
# show_data("conv1 weight",net.params['conv1'][0].data.reshape(32*3,5,5))
# #显示pool1的结果
# show_data("pool1 output", net.blobs['pool1'].data[0])
# #显示conv2的output
# show_data("conv2 output", net.blobs['conv2'].data[0], padval=0.5)
# show_data("conv2 weight",net.params['conv2'][0].data.reshape(32**2,5,5))
# #显示pool2的结果
# show_data("pool2 output", net.blobs['pool2'].data[0])
# #显示conv3的结果，weight显示前1024个
# show_data("conv3 output", net.blobs['conv3'].data[0], padval=0.5)
# show_data("conv3 weight",net.params['conv3'][0].data.reshape(64*32,5,5)[:1024])
# #显示pool3的结果
# show_data("pool3 output", net.blobs['pool3'].data[0], padval=0.2)

#ip1 第一个全连接层的输出
ip1=net.blobs['ip1'].data[0]
# print("ip1:", ip1, ip1.shape)
# plt.title("ip1")
# x=np.arange(len(ip1))
# plt.scatter(x, ip1.flat)
# plt.show()

#ip2 第二个全连接层的输出
# ip2=net.blobs['ip2'].data[0]
# print("ip2:", ip2)
# plt.title("ip2")
# x=np.arange(len(ip2))
# plt.scatter(x, ip2.flat)
# #plt.plot(ip2.flat)
# plt.show()

#最后输出层，为概率
labels = ['Airplane', 'AutoMobile', 'Bird', 'Cat', 'Deer', \
          'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
#prob=net.blobs['prob'].data[0]
prob = np.array(net.blobs['prob'].data[0])
print(prob.shape)
print("It is ", labels[np.argmax(prob)])
# print("prob \n")
for i in range(len(labels)):
    print(labels[i], prob[i])
# #print("prob", prob)

# plt.title("prob")
# x=np.arange(len(prob))
# plt.scatter(x, prob.flat)
# #plt.plot(prob.flat)
# plt.show()

