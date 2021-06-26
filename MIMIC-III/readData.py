
import os
import random
import math
import numpy as np
class ReadPhysionetData():
    # first read all dataset
    # before call, determine wheher shuffle
    # produce next batch
    def __init__(self, dataPath):
       

        r = np.load("./"+ dataPath +".npz")
        train_x = np.asarray(r[(dataPath+"x")])
        train_m = np.asarray(r[(dataPath+"m")])
        
        train_label=[]
        for x in r[(dataPath+'y')]:
            if x ==1:
                train_label.append([0,1])
            else:
                train_label.append([1,0])
        
        
        train_y = train_label
        train_x[train_m==0] = 0
        self.x=train_x.tolist()
        self.y=train_y
        self.times=[[60*j for j in range(48)] for i in range(len(self.x))]
        self.features=86
        self.m_train=train_m.tolist() # mask 0/1

        m=[] # mask 0/1
        m_mse=[]
        m_cover=[]

        x_cover=[]

        for onefile in self.x:
            one_m=[]
            one_m_cover=[]
            one_m_mse=[]

            shp = np.array(onefile).shape
            one_x_np_flat = np.array(onefile).reshape(-1)
            one_x_np_m = ~(one_x_np_flat==0)



            indices = np.where(~(one_x_np_flat==0))[0].tolist()
            indices = np.random.choice(indices, len(indices) // 10)

            onefile_cover = one_x_np_flat.copy()
            onefile_cover[indices] = 0.0
            onefile_m_cover = ~(onefile_cover==0)
            onefile_m_mse = (~(onefile_cover==0)) ^ (~(one_x_np_flat==0))

            # original m
            one_m = (one_x_np_m*1).reshape(shp).tolist()
            m.append(one_m)

            # final x
            onefile_cover = (onefile_cover).reshape(shp).tolist()
            x_cover.append(onefile_cover)

            # final m
            one_m_cover = (onefile_m_cover*1).reshape(shp).tolist()
            m_cover.append(one_m_cover)

            # mse m
            one_m_mse = (onefile_m_mse*1).reshape(shp).tolist()
            m_mse.append(one_m_mse)

        self.m=m
        self.x_=self.x.copy()
        self.x=x_cover
        self.m_cover=m_cover
        self.m_mse=m_mse

        x_lengths=[] #
        deltaPre=[] #time difference 
        lastvalues=[] # if missing, last values
        deltaSub=[]
        subvalues=[]


        for h in range(len(self.x)):
            # oneFile: steps*value_number
            oneFile=self.x[h] # 第h个文件
            one_time=self.times[h]# 第h个文件
            x_lengths.append(len(oneFile))
            
            one_deltaPre=[]
            one_lastvalues=[]
            
            one_deltaSub=[]
            one_subvalues=[]
            
            one_m=m[h]# 第h个文件
            for i in range(len(oneFile)):
                t_deltaPre=[0.0]*len(oneFile[i])
                t_lastvalue=[0.0]*len(oneFile[i])
                one_deltaPre.append(t_deltaPre)
                one_lastvalues.append(t_lastvalue)
                
                if i==0:
                    for j in range(len(oneFile[i])):
                        one_lastvalues[i][j]=0.0 if one_m[i][j]==0 else oneFile[i][j]
                    continue
                for j in range(len(oneFile[i])):
                    # 时间间隔
                    if one_m[i-1][j]==1:
                        one_deltaPre[i][j]=one_time[i]-one_time[i-1]
                    if one_m[i-1][j]==0:
                        one_deltaPre[i][j]=one_time[i]-one_time[i-1]+one_deltaPre[i-1][j]
                    # 上一个可见值
                    if one_m[i][j]==1:
                        one_lastvalues[i][j]=oneFile[i][j]
                    if one_m[i][j]==0:
                        one_lastvalues[i][j]=one_lastvalues[i-1][j]
        
            for i in range(len(oneFile)):
                t_deltaSub=[0.0]*len(oneFile[i])
                t_subvalue=[0.0]*len(oneFile[i])
                one_deltaSub.append(t_deltaSub)
                one_subvalues.append(t_subvalue)
            #construct array 
            for i in range(len(oneFile)-1,-1,-1):    
                if i==len(oneFile)-1:
                    for j in range(len(oneFile[i])):
                        one_subvalues[i][j]=0.0 if one_m[i][j]==0 else oneFile[i][j]
                    continue
                for j in range(len(oneFile[i])):
                    if one_m[i+1][j]==1:
                        one_deltaSub[i][j]=one_time[i+1]-one_time[i]
                    if one_m[i+1][j]==0:
                        one_deltaSub[i][j]=one_time[i+1]-one_time[i]+one_deltaSub[i+1][j]
                        
                    if one_m[i][j]==1:
                        one_subvalues[i][j]=oneFile[i][j]
                    if one_m[i][j]==0:
                        one_subvalues[i][j]=one_subvalues[i+1][j]   
                
            
            #m.append(one_m)
            deltaPre.append(one_deltaPre)
            lastvalues.append(one_lastvalues)
            deltaSub.append(one_deltaSub)
            subvalues.append(one_subvalues)



        # self.m=m
        self.deltaPre=deltaPre
        self.lastvalues=lastvalues
        self.deltaSub=deltaSub
        self.subvalues=subvalues
        self.x_lengths=x_lengths
        self.maxLength=max(x_lengths)

    
    def nextBatch(self):
        i=1
        while i*self.batchSize<=len(self.x):
            # 已求得的值
            x=[] # 时间切片之后的原始数据
            y=[] # 标签
            m=[] # missing矩阵
            deltaPre=[] # 从前往后的时间间隔
            x_lengths=[] # 原始x的长度
            lastvalues=[] # 前一个值
            deltaSub=[] # 从后往前的时间间隔
            subvalues=[] # 前一个值

            m_cover=[]
            m_mse=[]
            x_=[]

            imputed_deltapre=[]
            imputed_m=[]
            imputed_deltasub=[]
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                x.append(self.x[j])
                x_.append(self.x_[j])
                y.append(self.y[j])
                m.append(self.m[j])
                m_cover.append(self.m_cover[j])
                m_mse.append(self.m_mse[j])
                deltaPre.append(self.deltaPre[j])
                deltaSub.append(self.deltaSub[j])
                #放的都是引用，下面添加0，则原始数据也加了0
                x_lengths.append(self.x_lengths[j])
                lastvalues.append(self.lastvalues[j])
                subvalues.append(self.subvalues[j])
                jj=j-(i-1)*self.batchSize
                #times.append(self.times[j])
                while len(x[jj])<self.maxLength:
                    t1=[0.0]*(len(self.dic)-1)
                    x[jj].append(t1)
                    x_[jj].append(t1)
                    #times[jj].append(0.0)
                    t2=[0]*(len(self.dic)-1)
                    m[jj].append(t2)
                    m_cover[jj].append(t2)
                    m_mse[jj].append(t2)
                    t3=[0.0]*(len(self.dic)-1)
                    deltaPre[jj].append(t3)
                    t4=[0.0]*(len(self.dic)-1)
                    lastvalues[jj].append(t4)
                    t5=[0.0]*(len(self.dic)-1)
                    deltaSub[jj].append(t5)
                    t6=[0.0]*(len(self.dic)-1)
                    subvalues[jj].append(t6)

                #重新设置times,times和delta类似，但times生成的时候m全是1,用于生成器G
            i+=1
            '''
            x：data
            y: label
            m: mask
            deltaPre: time delta(frond to end)
            x_lengths: x_lengths
            lastvalues: imputed by last
            imputed_deltapre: generator time
            imputed_m: generator mask
            deltaSub: time delta(end to frond)
            subvalues: imputed by next
            imputed_deltasub: generator time
            '''

            yield  x,y,x_,m_cover,m_mse,m,deltaPre,x_lengths,lastvalues,deltaSub,subvalues

                
        
    def shuffle(self,batchSize=32,isShuffle=False):
        self.batchSize=batchSize
        if isShuffle:
            c = list(zip(self.x,self.x_,self.m_cover,self.m_mse,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.times,self.deltaSub,self.subvalues,self.m_train))
            random.shuffle(c)
            self.x,self.x_,self.m_cover,self.m_mse,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.times,self.deltaSub,self.subvalues,self.m_train=zip(*c)

if __name__ == '__main__':
    
    dt=ReadPhysionetData("test_")
    print(np.flatnonzero(dt.x[0]).shape)
    # print(dt.x[0].shape)
    print(np.shape(dt.x))
    print(np.shape(dt.x_))
    print(np.shape(dt.m))
    print(np.shape(dt.m_cover))
    print(np.shape(dt.m_mse))
    print(np.shape(dt.y))
    print(np.sum(np.array(dt.m)==1))
    print(np.sum(np.array(dt.m_cover)==1))
    print(np.sum(np.array(dt.m_mse)==1))
    print(np.sum(np.array(dt.m_train)==1))
    np.savez("./test", test_origin_x=dt.x_,test_x=dt.x,test_origin_m=dt.m,test_m=dt.m_cover,test_y=dt.y,m_mse=dt.m_mse)

