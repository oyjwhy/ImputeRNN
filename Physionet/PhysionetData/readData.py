# -*- coding: utf-8 -*-
"""
Created on 2021/06/26

@author: JiaWei OuYang
"""

import os
import random
import math
import numpy as np
import copy
import pickle
class ReadPhysionetData():
    # first read all dataset
    # before call, determine wheher shuffle
    # produce next batch
    def __init__(self, dataPath, labelPath,isNormal,isSlicing,sliceGap=60):
        print("data path: "+labelPath)
        labelFile = open(labelPath)
        fileNames=[]
        labels=[]
        #dataset: filenames,labels
        line_num = 0 
        for line in  labelFile.readlines():
        # rstrip() remove spaces in right end
            if line_num!=0:
                words = line.strip().split(',') 
                if os.path.isfile(os.path.join(dataPath, words[0]+".txt")):
                    fileNames.append(words[0]+".txt" )
                    if words[-1]=="0":
                        labels.append([1,0])
                    if words[-1]=="1":
                        labels.append([0,1])
            line_num=line_num+1
        self.dataPath = dataPath
        self.fileNames = fileNames
        labelFile.close()
        dic={'time':-1,'Age':0,'Gender':1,'Height':2,'ICUType':3,'Weight':4,'Albumin':5,\
             'ALP':6,'ALT':7,'AST':8,'Bilirubin':9,'BUN':10,'Cholesterol':11,'Creatinine':12,\
             'DiasABP':13,'FiO2':14,'GCS':15,'Glucose':16,'HCO3':17,'HCT':18,'HR':19,\
             'K':20,'Lactate':21,'Mg':22,'MAP':23,'MechVent':24,'Na':25,'NIDiasABP':26,\
             'NIMAP':27,'NISysABP':28,'PaCO2':29,'PaO2':30,'pH':31,'Platelets':32,'RespRate':33,\
             'SaO2':34,'SysABP':35,'Temp':36,'TroponinI':37,'TroponinT':38,'Urine':39,'WBC':40}
    
        self.dic=dic
        mean=[0.0]*(len(dic)-1)
        meancount=[0]*(len(dic)-1)
        x=[]
        times=[]
        non_in_dic_count=0
        # times: totalFilesLength*steps
        # x: totalFilesLength*steps*feature_length
        for fileName in fileNames:
            f=open(os.path.join(self.dataPath, fileName))
            count=0
            age=gender=height=icutype=weight=-1
            lastTime=0
            totalData=[]
            t_times=[]
            for line in f.readlines():
                if count > 1:
                    words=line.strip().split(",")
                    timestamp=words[0]
                    feature=words[1]
                    value=words[2]
                    
                    # 0 is missing value,orignl gender is 0/1 ,after preprocessing
                    # gender is 0/1/2(missing,male,female)
                    if timestamp == "00:00":
                        if feature=='Age':
                            age="0" if value=="-1" else value
                            #calcuate mean
                            if age !="0":
                                mean[self.dic[feature]]+=float(age)
                                meancount[self.dic[feature]]+=1
                        if feature=='Gender':
                            if value=="-1":
                                gender="0"
                            if value=="0":
                                gender="1"
                            if value=="1":
                                gender="2"
                            #calcuate mean
                            if gender !="0":
                                mean[self.dic[feature]]+=float(gender)
                                meancount[self.dic[feature]]+=1
                        if feature=='Height':
                            height="0" if value=="-1" else value
                            #calcuate mean
                            if height !="0":
                                mean[self.dic[feature]]+=float(height)
                                meancount[self.dic[feature]]+=1
                        if feature == 'ICUType':
                            icutype="0" if value=="-1" else value
                            #calcuate mean
                            if icutype !="0":
                                mean[self.dic[feature]]+=float(icutype)
                                meancount[self.dic[feature]]+=1
                        if feature=='Weight':
                            weight="0" if value=="-1" else value
                            #calcuate mean
                            if weight !="0":
                                mean[self.dic[feature]]+=float(weight)
                                meancount[self.dic[feature]]+=1
                    else:
                        if timestamp!=lastTime:
                            data=[0.0]*(len(dic)-1)
                            hourandminute=timestamp.split(":")
                            t_times.append(float(hourandminute[0])*60+float(hourandminute[1]))
                            data[0]=float(age)
                            data[1]=float(gender)
                            data[2]=float(height)
                            data[3]=float(icutype)
                            data[4]=float(weight)
                            
                            data[self.dic[feature]]=float(value)
                            mean[self.dic[feature]]+=float(value)
                            meancount[self.dic[feature]]+=1
                            
                            totalData.append(data)
                        else:
                            
                            totalData[-1][self.dic[feature]]=float(value)
                            mean[self.dic[feature]]+=float(value)
                            meancount[self.dic[feature]]+=1
                            
                            
                    lastTime=timestamp      
                count+=1
                #if len(totalData)==24:
                #    break;
            
            x.append(totalData)
            times.append(t_times)
            f.close()
       
        self.x=x
        self.y=labels
        self.times=times
       
        # print(np.array(self.times[0]).shape)
   
        self.timeslicing(isSlicing,sliceGap)
        
        
        for i in range(len(mean)):
            if meancount[i]!=0:
                mean[i]=mean[i]/meancount[i]
        self.mean=mean
        
        
        # normalization
        m=[] # mask 0/1
        # first calculate std
        self.std=[0.0]*(len(dic)-1)
        for onefile in self.x:
            one_m=[]
            for oneclass in onefile:
                t_m=[0]*len(oneclass)
                for j in range(len(oneclass)):
                    if oneclass[j] !=0:
                        self.std[j]+=(oneclass[j]-self.mean[j])**2
                        t_m[j]=1
                one_m.append(t_m)
            m.append(one_m)
        for j in range(len(self.std)):
            self.std[j]=math.sqrt(1.0/(meancount[j]-1)*self.std[j])
        
        self.isNormal=isNormal
        self.normalization(isNormal)    
        
        # print(np.array(m).shape)
                        
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
        self.m=m
        self.deltaPre=deltaPre
        self.lastvalues=lastvalues
        self.deltaSub=deltaSub
        self.subvalues=subvalues
        self.x_lengths=x_lengths
        self.maxLength=max(x_lengths)

        self.collect = list(zip(self.x,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.fileNames,self.times,self.deltaSub,self.subvalues))

        print("max_length is : "+str(self.maxLength))
        print("non_in_dic_count is : "+str(non_in_dic_count))
        
        # resultFile=open(os.path.join("./Physionet2012Data","meanAndstd"),'w')
        resultFile=open(os.path.join("./","meanAndstd"),'w')
        for i in range(len(self.mean)):
            resultFile.writelines(str(self.mean[i])+","+str(self.std[i])+","+str(meancount[i])+"\r")
        resultFile.close()
        
    def normalization(self,isNormal):
        if not isNormal:
            return
        for onefile in self.x:
            for oneclass in onefile:
                for j in range(len(oneclass)):
                    if oneclass[j] !=0:
                        if self.std[j]==0:
                            oneclass[j]=0.0
                        else:
                            oneclass[j]=1.0/self.std[j]*(oneclass[j]-self.mean[j])

    
    # 时间切片
    def timeslicing(self,isSlicing,sliceGap):
        #slicing x, make time gap be 30min, get the average of 30min
        if not isSlicing:
            return
        else:
            newx=[]
            newtimes=[]
            # 每个i是一个文件，需要对文件中的时间规整化
            for i in range(len(self.times)):
                nowx=self.x[i] # [..., ...]
                nowtime=self.times[i] # [...]
                lasttime=0
                newnowx=[]
                newnowtime=[]
                count=[0.0]*(len(self.dic)-1)
                tempx=[0.0]*(len(self.dic)-1)
                #newnowx.append(tempx)
                #newnowtime.append(lasttime)
                nowtime.append(48*60+2)
                for j in range(len(nowtime)):
                    if nowtime[j]<=lasttime+sliceGap:
                        for k in range(0,len(self.dic)-1):
                            tempx[k]+=nowx[j][k]
                            if nowx[j][k]!=0:
                                count[k]+=1.0
                    else:
                        for k in range(0,len(self.dic)-1):
                            if count[k]==0:
                                count[k]=1.0
                            tempx[k]=tempx[k]/count[k] # 求均值
                        while nowtime[j]>lasttime+sliceGap: # 可能nowtime[j]跨度太长
                            newnowx.append(tempx)
                            newnowtime.append(lasttime)
                            lasttime+=sliceGap
                            count=[0.0]*(len(self.dic)-1)
                            tempx=[0.0]*(len(self.dic)-1)
                        # j may be len(nowx), we add one point into nowtime before
                        # 当前所在时间点的值得加进去
                        if j<len(nowx):
                            for k in range(0,len(self.dic)-1):
                                tempx[k]+=nowx[j][k]
                                if nowx[j][k]!=0:
                                    count[k]+=1.0
                
                # while循环内的直接赋值0的完善操作
                for j in range(len(newnowtime)):
                    if newnowx[j][0]==0:
                        newnowx[j][0]=nowx[0][0]
                    if newnowx[j][1]==0:
                        newnowx[j][1]=nowx[0][1]
                    if newnowx[j][2]==0:
                        newnowx[j][2]=nowx[0][2]
                    if newnowx[j][3]==0:
                        newnowx[j][3]=nowx[0][3]
                    if newnowx[j][4]==0:
                        newnowx[j][4]=nowx[0][4]
                            
                nowtime.pop(-1)
                newx.append(newnowx)
                newtimes.append(newnowtime)
            self.x=newx
            self.times=newtimes
    
    
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

            imputed_deltapre=[]
            imputed_m=[]
            imputed_deltasub=[]
            mean=self.mean
            files=[]
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                files.append(self.fileNames[j])
                x.append(self.x[j])
                y.append(self.y[j])
                m.append(self.m[j])
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
                    #times[jj].append(0.0)
                    t2=[0]*(len(self.dic)-1)
                    m[jj].append(t2)
                    t3=[0.0]*(len(self.dic)-1)
                    deltaPre[jj].append(t3)
                    t4=[0.0]*(len(self.dic)-1)
                    lastvalues[jj].append(t4)
                    t5=[0.0]*(len(self.dic)-1)
                    deltaSub[jj].append(t5)
                    t6=[0.0]*(len(self.dic)-1)
                    subvalues[jj].append(t6)

            # 为生成器准备数据
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                one_imputed_deltapre=[]
                one_imputed_deltasub=[]
                one_G_m=[]
                for h in range(0, self.x_lengths[j]):
                    if h==0:
                        one_f_time=[0.0]*(len(self.dic)-1)
                        one_imputed_deltapre.append(one_f_time)
                        try:
                            one_sub=[self.times[j][h+1]-self.times[j][h]]*(len(self.dic)-1) # 数组内值为时间差值
                        except:
                            print("error: "+str(h)+" "+str(len(self.times[j]))+" "+self.fileNames[j])
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(len(self.dic)-1)
                        one_G_m.append(one_f_g_m)
                    elif h==self.x_lengths[j]-1:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*(len(self.dic)-1)
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[0.0]*(len(self.dic)-1)
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(len(self.dic)-1)
                        one_G_m.append(one_f_g_m)
                    else:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*(len(self.dic)-1)
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[self.times[j][h+1]-self.times[j][h]]*(len(self.dic)-1)
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(len(self.dic)-1)
                        one_G_m.append(one_f_g_m)
                while len(one_imputed_deltapre)<self.maxLength:
                    one_f_time=[0.0]*(len(self.dic)-1)
                    one_imputed_deltapre.append(one_f_time)
                    one_sub=[0.0]*(len(self.dic)-1)
                    one_imputed_deltasub.append(one_sub)
                    one_f_g_m=[0.0]*(len(self.dic)-1)
                    one_G_m.append(one_f_g_m)
                imputed_deltapre.append(one_imputed_deltapre)
                imputed_deltasub.append(one_imputed_deltasub)
                imputed_m.append(one_G_m)
                #重新设置times,times和delta类似，但times生成的时候m全是1,用于生成器G
            i+=1
            '''
            x：data
            y: label
            mean: mean
            m: mask
            deltaPre: time delta(frond to end)
            x_lengths: x_lengths
            lastvalues: imputed by last
            files: files
            imputed_deltapre: generator time
            imputed_m: generator mask
            deltaSub: time delta(end to frond)
            subvalues: imputed by next
            imputed_deltasub: generator time
            '''


            if self.isNormal:
                yield  x,y,[0.0]*(len(self.dic)-1),m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
            else:
                yield  x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
                
        
    def shuffle(self,batchSize=32,isShuffle=False):
        self.batchSize=batchSize
        if isShuffle:
            random.shuffle(self.collect)
            self.x,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.fileNames,self.times,self.deltaSub,self.subvalues=zip(*self.collect)

    def crossValidation(self, foldth):

        trainlistorder = list(range(0,foldth*400)) + list(range(foldth*400 + 396,3596))
        validlistorder_ = list(range(foldth*400,foldth*400 + 396))
        listorder = trainlistorder + validlistorder_
        c = [self.collect[i][:] for i in listorder]
        self.x,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.fileNames,self.times,self.deltaSub,self.subvalues=zip(*c)


if __name__ == '__main__':

    dt=ReadPhysionetData("../../set-a/train", "../../set-a/train/list.txt",isNormal=True,isSlicing=True)
    print(np.sum(np.asarray(dt.x)==0))
    print(np.sum(np.asarray(dt.m)==0))
