# -*- coding: utf-8 -*-
"""
Created on 2021/06/26

@author: JiaWei OuYang
"""
from __future__ import print_function
import imputeGRU 
import tensorflow as tf
import argparse
import readData

"""main"""
def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-path', type=str, default="")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss-weight',type=int,default=0.1)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--n-inputs', type=int, default=86)
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

    dt_train=readData.ReadPhysionetData("train_")
    dt_test=readData.ReadPhysionetData("test_")

    maxlr = 0
    maxlw = 0

    lrs=[0.0012]
    loss_weight = [0.001]
    for maxlr in lrs:
        for maxlw in loss_weight:
            f=open("MAX_AUC","a")
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

            with tf.Session(config=config) as sess:
                    aGru = imputeGRU.imputegru(sess,
                            args=args,
                            dataset=dt_train,
                            testset=dt_test,)

                    aGru.build()
                    aGru.test()
                    print(" [*] Test dataset Imputation finished!")

            tf.reset_default_graph()
if __name__ == '__main__':
    main()
