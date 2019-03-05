#!/usr/bin/env python
import matplotlib.pyplot as plt
file =open('quick1.log.train')
filelines=file.readlines()
#print(len(filelines))
Iters=[]
Seconds=[]
TrainingLoss=[]
temp=[]
for i  in range (1,len(filelines)):
    line=filelines[i].split(' ')
    #print line
    for j in range(0,len(line)):
        if line[j] !='':
           #print line[j]
           temp.append(line[j])
 
#print(len(temp))
for i in range(0,len(temp)):
    if i%4==0:
        Iters.append(int(temp[i]))
    if i%4==1:
        Seconds.append(float(temp[i]))

 
    if i%4==2:
        TrainingLoss.append(float(temp[i]))
 

plt.plot(Iters, TrainingLoss, 'b')
plt.title('Trainloss  VS Iters')
plt.xlabel('Iters')
#plt.plot(Seconds, TrainingLoss, 'r*-')

plt.plot(Seconds, TrainingLoss, 'r-',marker='d')
plt.title('Trainloss  VS Seconds')
plt.xlabel('Seconds')
 
plt.ylabel('Trainloss')
#plt.savefig('trainloss.png')
plt.show()
