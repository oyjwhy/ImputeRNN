# -*- coding: utf-8 -*-
"""
Created on 2021/06/26

@author: JiaWei OuYang
"""

from __future__ import print_function
from PhysionetData import readData, readTestData
import imputeGRU 
import tensorflow as tf
import argparse
import pickle
import os
import time

"""main"""
def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-path', type=str, default="./set-a/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss-weight',type=int,default=0.1)
    parser.add_argument('--epoch', type=int, default=101)
    parser.add_argument('--n-inputs', type=int, default=41)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--isNormal',type=int,default=1)
    parser.add_argument('--isBatch-normal',type=int,default=1)
    parser.add_argument('--isSlicing',type=int,default=1)
    parser.add_argument('--foldth',type=int,default=0)
    args = parser.parse_args()
    
    if args.isBatch_normal==0:
            args.isBatch_normal=False
    if args.isBatch_normal==1:   
            args.isBatch_normal=True
    if args.isNormal==0:
            args.isNormal=False
    if args.isNormal==1:
            args.isNormal=True
    if args.isSlicing==0:
            args.isSlicing=False
    if args.isSlicing==1:
            args.isSlicing=True

    # train set
    if os.path.isfile('./PhysionetData/data_train.obj'):
        print("Read TrainSet from obj file")
        fileObj = open('./PhysionetData/data_train.obj', 'rb')
        dt_train = pickle.load(fileObj)
        fileObj.close()
    else:
        print("Read TrainSet from set-a")
        dt_train=readData.ReadPhysionetData(os.path.join(args.data_path,"train"),\
        os.path.join(args.data_path,"train","list.txt"),\
        isNormal=args.isNormal,isSlicing=args.isSlicing)

        # save time for reading train set
        fileObj = open('./PhysionetData/data_train.obj', 'wb')
        pickle.dump(dt_train, fileObj)
        fileObj.close()

    # test set
    print("Read TestSet from set-a")
    dt_test=readTestData.ReadPhysionetData(os.path.join(args.data_path,"test"),\
    os.path.join(args.data_path,"test","list.txt"),\
    dt_train.maxLength,isNormal=args.isNormal,isSlicing=args.isSlicing)


    maxlr = 0
    maxlw = 0

    lrs=[0.01]
    loss_weight = [0.0001]
    for maxlr in lrs:
        for maxlw in loss_weight:
            f=open("MAX_AUC","a")
            f.write("\n===================\n"+str(maxlr)+":"+str(maxlw)+"\n===================\n")
            f.close()
            f=open("MIN_RMSE_MAE","a")
            f.write("\n===================\n"+str(maxlr)+":"+str(maxlw)+"\n===================\n")
            f.close()
            print("================Test Model=======================")
            print("Selected parameters: lr[%2.4f] and lw[%2.4f]" % (maxlr, maxlw))
            print("=================================================")
            
            args.lr = maxlr
            args.loss_weight = maxlw
            tf.reset_default_graph()
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True 
            
            time_start=time.time()
            with tf.Session(config=config) as sess:
                    aGru = imputeGRU.imputegru(sess,
                            args=args,
                            dataset=dt_train,
                            testset=dt_test,)

                    aGru.build()
                    aGru.test()
                    print(" [*] Test dataset Imputation finished!")
            time_end=time.time()
            print('totally cost',(time_end-time_start) * 1000)
            tf.reset_default_graph()

if __name__ == '__main__':
    main()
