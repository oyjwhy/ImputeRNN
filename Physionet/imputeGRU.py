from __future__ import print_function
import os
import numpy as np
from sklearn import metrics
import time
import mygru_cell
import tensorflow as tf
from tensorflow.python.ops import math_ops,array_ops
tf.set_random_seed(1)   # set random seed
 
class imputegru(object):
    model_name = "imputeGRU"
    def __init__(self, sess, args, dataset, testset):
        self.lr = args.lr         
        self.loss_weight = args.loss_weight   
        self.sess=sess
        self.isbatch_normal=args.isBatch_normal
        self.isNormal=args.isNormal
        self.isSlicing=args.isSlicing
        self.dataset=dataset
        self.epoch = args.epoch     
        self.batch_size = args.batch_size
        self.n_inputs = args.n_inputs
        self.n_steps = dataset.maxLength                                # time steps
        self.n_hidden_units = args.n_hidden_units        # neurons in hidden layer
        self.n_classes = args.n_classes
        self.log_dir=args.log_dir
        self.checkpoint_dir=args.checkpoint_dir
        self.foldth=args.foldth
        self.num_batches = len(dataset.x) // self.batch_size

        # x y placeholder
        self.keep_prob = tf.placeholder(tf.float32) 
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.x_ = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.m_mse = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.delta = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.lastvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.x_lengths = tf.placeholder(tf.int32,  shape=[self.batch_size,])
        self.diagnoal_constant = tf.constant(1., shape=[self.n_inputs, self.n_inputs]) - tf.matrix_diag(tf.constant(1., shape=[self.n_inputs, ]))
        self.testset = testset


    def RNN(self,X, M, Delta, Lastvalues, X_lengths,Keep_prob, reuse=False):
        
         with tf.variable_scope("imputegru", reuse=reuse):
            w_out=tf.get_variable('w_out', shape=[self.n_hidden_units, self.n_classes],initializer=tf.random_normal_initializer())
            b_out=tf.get_variable('b_out', shape=[self.n_classes, ],initializer=tf.constant_initializer(0.001))
            wr_h=tf.get_variable('wr_h',shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            br_h=tf.get_variable('br_h', shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))

            M=tf.reshape(M,[-1,self.n_inputs])
            X = tf.reshape(X, [-1, self.n_inputs])
            Delta=tf.reshape(Delta,[-1,self.n_inputs])
            rth= tf.matmul(Delta, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))

            X=tf.concat([X,M,rth],1)
            X_in = tf.reshape(X, [-1, self.n_steps, self.n_inputs*2+self.n_hidden_units])


            imputegru_cell = mygru_cell.MyGRUCell15(self.n_hidden_units * 2, self.n_inputs, reuse=tf.AUTO_REUSE)
            imputegru_cell_bw = mygru_cell.MyGRUCell15(self.n_hidden_units * 2, self.n_inputs, reuse=tf.AUTO_REUSE)


            init_state = imputegru_cell.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            init_state_back = imputegru_cell_bw.zero_state(self.batch_size, dtype=tf.float32)


            # return results,outputs,final_state
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(imputegru_cell, imputegru_cell_bw, X_in, \
                                initial_state_fw=init_state,
                                initial_state_bw=init_state_back,
                                sequence_length=X_lengths,
                                time_major=False)

            states_fw, states_bw = final_state
            state = ((states_fw + states_bw)/2)[:, 0:self.n_hidden_units]
            outputs_fw, outputs_bw = outputs
            output = (outputs_fw + outputs_bw)/2

            tempout=tf.matmul(tf.nn.dropout(state, Keep_prob), w_out) + b_out 
            results =tf.nn.softmax(tempout)   #选取最后一个 output

            return results,output,final_state



    def build(self):
        
        self.pred,self.outputs,self.final_state = self.RNN(self.x, self.m, self.delta, self.lastvalues, self.x_lengths, self.keep_prob)
        self.rmse = tf.reduce_sum(tf.square(tf.multiply((self.outputs - self.x_), self.m_mse))) / tf.reduce_sum(self.m_mse)
        self.mae = tf.reduce_sum(tf.abs(tf.multiply((self.outputs - self.x_), self.m_mse))) / tf.reduce_sum(self.m_mse)

        self.cross_entropy = -tf.reduce_sum(self.y*tf.log(self.pred))
        self.impute_loss = tf.reduce_sum(tf.square(tf.multiply((self.outputs - self.x), self.m))) / tf.reduce_sum(self.m)
        self.combine_loss = self.cross_entropy  * self.loss_weight + self.impute_loss

        # self.rmse = tf.reduce_sum(tf.square(tf.multiply((self.outputs - self.x), self.m))) / tf.reduce_sum(self.m)
        # self.mae = tf.reduce_sum(tf.abs(tf.multiply((self.outputs - self.x), self.m))) / tf.reduce_sum(self.m)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.combine_loss)
         
        
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.saver = tf.train.Saver()
        
        loss_sum = tf.summary.scalar("loss", self.cross_entropy)
        acc_sum = tf.summary.scalar("acc", self.accuracy)
        
        self.sum=tf.summary.merge([loss_sum, acc_sum])
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)
        
        
    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}/epoch{}_foldth{}".format(
            self.model_name, self.lr,self.loss_weight,
            self.batch_size, self.isNormal,
            self.isbatch_normal,self.isSlicing,
            self.epoch, self.foldth
            )
        
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if not step==0:
            self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    
    def train(self):
        
        max_auc = 0
        max_epoch = 0
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            idx=start_batch_id
            counter=start_epoch*self.num_batches
            epochcount=start_epoch
            print(" [*] Load SUCCESS")
        else:
            # initialize all variables
            tf.global_variables_initializer().run()
            epochcount=0
            counter = 0
            idx = 0
            print(" [!] Load failed...")
        start_time=time.time()
        dataset=self.dataset
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # dataset.shuffle(self.batch_size,True)
        while epochcount<self.epoch:
            counter = 0
            next_genetor = dataset.nextBatch()
            for i in range(0,25):

                data_x,data_y,data_mean,data_m,data_delta,data_x_lengths,data_lastvalues,_,_,_,_,_,_ = next(next_genetor)
                _,cross_entropy_loss,impute_loss,combine_loss,summary_str,acc,pred = self.sess.run([self.train_op,self.cross_entropy, self.impute_loss, self.combine_loss,self.sum, self.accuracy,self.pred], feed_dict={\
                    self.x: data_x,\
                    self.y: data_y,\
                    self.m: data_m,\
                    self.delta: data_delta,\
                    self.x_lengths: data_x_lengths,\
                    self.lastvalues: data_lastvalues,\
                    self.keep_prob: 0.5})
        
                self.writer.add_summary(summary_str, counter)
                counter += 1
                idx+=1
                if counter%10==0 or counter==25:
                    try:
                        auc = metrics.roc_auc_score(np.array(data_y),np.array(pred))
                    except ValueError:
                        pass

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, {y_loss: %.4f | impute_loss: %.4f | combine_loss: %.4f}, acc: %.8f , auc: %.8f" \
                            % (epochcount, idx, self.num_batches, time.time() - start_time, cross_entropy_loss, impute_loss, combine_loss, acc, auc))

                start_time=time.time()
                acccounter=0
                totalacc=0.0
                totalauc=0.0
                auccounter=0

            for data_x,data_y,data_mean,data_m,data_delta,data_x_lengths,data_lastvalues,_,_,_,_,_,_ in next_genetor:
                summary_str,acc,pred = self.sess.run([self.sum, self.accuracy,self.pred], feed_dict={\
                    self.x: data_x,\
                    self.y: data_y,\
                    self.m: data_m,\
                    self.delta: data_delta,\
                    self.x_lengths: data_x_lengths,\
                    self.lastvalues: data_lastvalues,\
                    self.keep_prob: 1.0})
        
                self.writer.add_summary(summary_str, acccounter)
                try:
                    auc = metrics.roc_auc_score(np.array(data_y),np.array(pred))
                    totalauc+=auc
                    auccounter+=1
                except ValueError:
                    pass
                totalacc+=acc
                acccounter += 1
                print("Batch: %4d time: %4.4f, acc: %.8f, auc: %.8f" \
                            % ( acccounter, time.time() - start_time, acc, auc))
            
            totalacc=totalacc/acccounter
            totalauc=totalauc/auccounter
            print("Total acc: %.8f, Total auc: %.8f , acccounter is : %.2f , auccounter is %.2f" % (totalacc,totalauc,acccounter,auccounter))

            epochcount+=1
            idx=0

            if totalauc > max_auc:
                max_auc = totalauc
                max_epoch = epochcount-1

            self.save(self.checkpoint_dir, epochcount)
        return max_auc,max_epoch

    def test(self):

        max_auc = 0
        max_epoch = 0
        min_rmse = 1000
        min_mae = 1000
        start_time=time.time()
        idx = 0
        epochcount=0
        dataset=self.dataset
        testdataset=self.testset
        tf.global_variables_initializer().run()
        save_imputation_=[]
        save_imputation_y=[] 
        save_test_imputation_=[]
        save_test_imputation_y=[] 
        imputed = []
        while epochcount<self.epoch:
            counter = 0
            epochcount+=1
            dataset.shuffle(self.batch_size,True)
            for data_x,data_y,data_mean,data_m,data_delta,data_x_lengths,data_lastvalues,_,_,_,_,_,_ in dataset.nextBatch():
                _,cross_entropy_loss,impute_loss,combine_loss,summary_str,acc,pred,imputed_outputs = self.sess.run([self.train_op,self.cross_entropy, self.impute_loss, self.combine_loss,self.sum, self.accuracy,self.pred,self.outputs], feed_dict={\
                    self.x: data_x,\
                    self.y: data_y,\
                    self.m: data_m,\
                    self.delta: data_delta,\
                    self.x_lengths: data_x_lengths,\
                    self.lastvalues: data_lastvalues,\
                    self.keep_prob: 0.5})
                counter += 1
                if counter%10==0 or counter==28:
                    try:
                        auc = metrics.roc_auc_score(np.array(data_y),np.array(pred))
                    except ValueError:
                        pass

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, {y_loss: %.4f | impute_loss: %.4f | combine_loss: %.4f}, acc: %.8f , auc: %.8f" \
                              % (epochcount, counter, self.num_batches, time.time() - start_time, cross_entropy_loss, impute_loss, combine_loss, acc, auc))
                
            #     if epochcount % 10 == 0:
            #         save_imputation_y.extend(data_y)
            #         imputed=np.array(1-np.array(data_m)) * np.array(imputed_outputs) + np.array(data_m) * np.array(data_x)
            #         save_imputation_.extend(imputed.tolist())

            # if epochcount % 10 == 0:
            #     np.savez("./train_gru"+str(self.lr)+str(epochcount), train_gru=save_imputation_, train_gru_y=save_imputation_y)
            #     save_imputation_ =[]
            #     save_imputation_y =[]
            
            start_time=time.time()
            acccounter=0
            totalacc=0.0
            totalauc=0.0
            totalrmse=0.0
            totalmae=0.0
            auccounter=0
            testdataset.shuffle(self.batch_size,False)
            for data_x,data_y,data_x_,data_m_cover,data_m_mse,data_mean,data_m,data_delta,data_x_lengths,data_lastvalues,_,_,_, in testdataset.nextBatch():
                acc,pred,rmse,mae,imputed_outputs = self.sess.run([self.accuracy,self.pred,self.rmse,self.mae,self.outputs], feed_dict={\
                    self.x: data_x,\
                    self.y: data_y,\
                    self.x_: data_x_,\
                    self.m_mse: data_m_mse,\
                    self.m: data_m_cover,\
                    self.delta: data_delta,\
                    self.x_lengths: data_x_lengths,\
                    self.lastvalues: data_lastvalues,\
                    self.keep_prob: 1.0})
                try:
                    auc = metrics.roc_auc_score(np.array(data_y),np.array(pred))
                    totalauc+=auc
                    auccounter+=1
                except ValueError:
                    pass
                totalacc+=acc
                totalrmse+=rmse
                totalmae+=mae
                acccounter += 1
                print("Batch: %4d time: %4.4f, acc: %.8f, auc: %.8f, rmse: %.8f, mae: %.8f" \
                            % ( acccounter, time.time() - start_time, acc, auc, rmse, mae))
            #     if epochcount % 10 == 0:
            #         save_test_imputation_y.extend(data_y)
            #         imputed=np.array(1-np.array(data_m_cover)) * np.array(imputed_outputs) + np.array(data_m_cover) * np.array(data_x)
            #         save_test_imputation_.extend(imputed.tolist())

            # if epochcount % 10 == 0:
            #     np.savez("./test_gru"+str(self.lr)+str(epochcount), test_gru=save_test_imputation_, test_gru_y=save_test_imputation_y)
            #     save_test_imputation_ =[]
            #     save_test_imputation_y =[]

            totalacc=totalacc/acccounter
            totalauc=totalauc/auccounter
            totalrmse=totalrmse/acccounter
            totalmae=totalmae/acccounter
            print("Total acc: %.8f, Total auc: %.8f , acccounter is : %.2f , auccounter is %.2f, totalrmse is : %.8f , totalmae is %.8f" \
                % (totalacc,totalauc,acccounter,auccounter,totalrmse,totalmae))

            if totalrmse <  min_rmse:
                min_rmse = totalrmse
            if totalmae < min_mae:
                min_mae = totalmae
            if totalauc > max_auc:
                max_auc = totalauc
                max_epoch = epochcount-1

        print("Max auc [%.8f] in Epoch [%2d]" % (max_auc, max_epoch))
        f=open("MAX_AUC","a")
        f.write(str(max_epoch)+":"+str(max_auc)+"\n")
        f.close()
        print("min_rmse [%.8f] min_mae [%.8f]" % (min_rmse, min_mae))
        f=open("MIN_RMSE_MAE","a")
        f.write(str(min_rmse)+":\t"+str(min_mae)+"\n")
        f.close()
        return totalacc,totalauc