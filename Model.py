# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import argparse
from DataSet import DataSet
import sys
import os
import heapq
import math
import matplotlib.pyplot as plt
import time

class BaseClass(object):
    def __init__(self,args):
        self.dataName = args.dataName
        self.dataSet = DataSet(self.dataName,args.isRate,args.lineSpliter,args.minId)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate

        self.train = self.dataSet.train
        self.test = self.dataSet.test

        self.negNum = args.negNum
        self.testNeg = self.dataSet.getTestNeg(self.test, 99)

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop

        self.checkPoint = args.checkPoint
        self.lr = args.lr
    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)
    def add_loss(self):
        pass

    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        # if os.path.exists(self.checkPoint):
        #     [os.remove(f) for f in os.listdir(self.checkPoint)]
        # else:
        #     os.mkdir(self.checkPoint)
    # def load_para(self):
    #     vari_list = 
    def run(self):
        print('='*20+'model Info'+'='*20)
        print(self.modelName)
        best_hr = -1
        best_NDCG = -1
        best_epoch = -1
        print("Start Training!")
        hr_list=[]
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            t1= time.time()
            self.run_epoch()
            t2= time.time()
            print('='*50)
            print("Start Evaluation!")
            self.evaluate(epoch)
            t3 = time.time()
            print("Epoch ", epoch, ' [',t2-t1,'] ',"HR: {}, NDCG: {}, [{}] ".format(self.hr, self.NDCG,t3-t2))
            hr_list.append(self.hr)
            # plt.scatter(epoch,self.hr,c=self.color)
            if self.hr > best_hr or self.NDCG > best_NDCG:
                best_hr = self.hr
                best_NDCG = self.NDCG
                best_epoch = epoch
                self.saver.save(self.sess, self.checkPoint)
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
        print("="*20+"Epoch ", epoch, "End"+"="*20)    
        print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
        print("Training complete!")
        plt.plot(hr_list,label=self.modelName)

    def run_epoch(self, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            _, tmp_loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}

    def evaluate(self,epoch):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0


        hr =[]
        NDCG = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = self.sess.run(self.y_, feed_dict=feed_dict)

            item_score_dict = {}
            lenth = len(testItem[i])
            for j in range(lenth):
                j = (j+1) % lenth
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            ranklist = heapq.nlargest(self.topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        self.hr, self.NDCG = np.mean(hr), np.mean(NDCG)

class DMF(BaseClass):
    def __init__(self, args):
        super(DMF,self).__init__(args)
        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.checkPoint = args.checkPoint+'DMF/'

        self.add_embedding_matrix()
        self.add_placeholders()

        self.add_model()
        self.add_loss()

        self.add_train_step()
        self.init_sess()

    def add_embedding_matrix(self):
        self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding())
        self.item_user_embedding = tf.transpose(self.user_item_embedding)

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))
            # user_out = tf.nn.relu(user_out)
        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))
            # item_out = tf.nn.relu(item_out)
        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (norm_item_output* norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

    def add_loss(self):
        regRate = self.rate / self.maxRate
        losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=regRate , logits= self.y_)
        loss = -tf.reduce_sum(losses)
        # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # self.loss = loss + self.reg * regLoss
        self.loss = loss

class GMF(BaseClass):
    def __init__(self, args,GMFlayer,MLPlayer,color):
        super(GMF,self).__init__(args)
        self.modelName = 'GMF'
        # self.MLPlayer = args.MLPlayer
        self.GMFlayer = GMFlayer
        self.color = color
        self.checkPoint = args.checkPoint+'neuMF/GMF'
        self.add_placeholders()
        self.add_model()
        self.add_loss()
        self.add_train_step()
        self.init_sess()
    def add_model(self):
        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)
        with tf.name_scope('GMF'):
            embed_u = init_variable([self.shape[0],self.GMFlayer],'embed_u')
            embed_i = init_variable([self.shape[1],self.GMFlayer],'embed_i')
            user_input = tf.nn.embedding_lookup(embed_u,self.user)
            item_input = tf.nn.embedding_lookup(embed_i,self.item)
            gmf_v = tf.multiply(user_input,item_input)
        with tf.name_scope('predict'):
            w_out = init_variable([self.GMFlayer,1],'w_out')
            b_out = init_variable([1],'b_out')
            self.y_ = tf.add(tf.matmul(gmf_v ,w_out),b_out)
            self.y_ = tf.reshape(self.y_,[-1])
    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rate/self.maxRate ,logits=self.y_))

class MLP(BaseClass):
    def __init__(self, args,GMFlayer,MLPlayer,color):
        super(MLP,self).__init__(args)
        self.modelName = 'MLP'
        self.MLPlayer = MLPlayer
        # self.GMFlayer = args.GMFlayer
        self.color = color
        self.checkPoint = args.checkPoint+'neuMF/MLP'
        self.add_placeholders()
        self.add_model()
        self.add_loss()
        self.add_train_step()
        self.init_sess()
    def add_model(self):
        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)
        with tf.name_scope('MLP'):
            embed_u = init_variable([self.shape[0],self.MLPlayer[0]//2],'embed_u')
            embed_i = init_variable([self.shape[1],self.MLPlayer[0]//2],'embed_i')
            user_input = tf.nn.embedding_lookup(embed_u,self.user)
            item_input = tf.nn.embedding_lookup(embed_i,self.item)
            mlp_v = tf.concat([user_input,item_input],1)
            for i in range(len(self.MLPlayer)-1):
                w = init_variable([self.MLPlayer[i],self.MLPlayer[i+1]],'w'+str(i+1))
                b = init_variable([self.MLPlayer[i+1]],'b'+str(i+1))
                mlp_v = tf.add(tf.matmul(mlp_v,w),b)
                mlp_v = tf.nn.relu(mlp_v)
        with tf.name_scope('predict'):
            w_out = init_variable([self.MLPlayer[-1],1],'w_out')
            b_out = init_variable([1],'b_out')
            self.y_ = tf.add(tf.matmul(mlp_v ,w_out),b_out)
            self.y_ = tf.reshape(self.y_,[-1])
    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rate/self.maxRate ,logits=self.y_))

class neuMF(BaseClass):
    def __init__(self, args,GMFlayer,MLPlayer,color):
        super(neuMF,self).__init__(args)
        self.modelName = 'neuMF'
        self.MLPlayer = MLPlayer
        self.GMFlayer = GMFlayer
        self.color = color
        self.checkPoint = args.checkPoint+'neuMF/neuMF'
        self.add_placeholders()
        self.add_model()
        self.add_loss()
        self.add_train_step()
        self.init_sess()
    def add_model(self):
        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)
        with tf.name_scope('MLP'):
            embed_u = init_variable([self.shape[0],self.MLPlayer[0]//2],'embed_u')
            embed_i = init_variable([self.shape[1],self.MLPlayer[0]//2],'embed_i')
            user_input = tf.nn.embedding_lookup(embed_u,self.user)
            item_input = tf.nn.embedding_lookup(embed_i,self.item)
            mlp_v = tf.concat([user_input,item_input],1)
            for i in range(len(self.MLPlayer)-1):
                w = init_variable([self.MLPlayer[i],self.MLPlayer[i+1]],'w'+str(i+1))
                b = init_variable([self.MLPlayer[i+1]],'b'+str(i+1))
                mlp_v = tf.nn.relu(tf.add(tf.matmul(mlp_v,w),b))
        with tf.name_scope('GMF'):
            embed_u = init_variable([self.shape[0],self.GMFlayer],'embed_u')
            embed_i = init_variable([self.shape[1],self.GMFlayer],'embed_i')
            user_input = tf.nn.embedding_lookup(embed_u,self.user)
            item_input = tf.nn.embedding_lookup(embed_i,self.item)
            gmf_v = tf.multiply(user_input,item_input)
        with tf.name_scope('predict'):
            concat_v = tf.concat([gmf_v,mlp_v],1)
            w_out = init_variable([self.GMFlayer+self.MLPlayer[-1],1],'w_out')
            b_out = init_variable([1],'b_out')
            self.y_ = tf.add(tf.matmul(concat_v ,w_out),b_out)
            self.y_ = tf.reshape(self.y_,[-1])
    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rate/self.maxRate ,logits=self.y_))

class co_GMF(object):
    def __init__(self,args):

    def add_placeholder(self):
        self.user_s = tf.placeholder(tf.int32)
        self.item_s = tf.placeholder(tf.int32)
        self.rate_s = tf.placeholder(tf.float32)
        self.user_t = tf.placeholder(tf.int32)
        self.item_t = tf.placeholder(tf.int32)
        self.rate_t = tf.placeholder(tf.float32)

    def add_model(self):
        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)
        with tf.name_scope('GMF'):
            embed_u_s = init_variable([self.shape_s[0],self.GMFlayer],'embed_u_s')
            embed_i_s = init_variable([self.shape_s[1],self.GMFlayer],'embed_i_s')
            embed_u_t = init_variable([self.shape_t[0],self.GMFlayer],'embed_u_t')
            embed_i_t = init_variable([self.shape_t[1],self.GMFlayer],'embed_i_t')
            alpha_e = tf.get_variable(shape=[self.shape_s[0],1],initializer=tf.ones_initializer,shape='alpha_e')
            belt_e = tf.get_variable(shape=[self.shape_t[0],1],initializer=tf.ones_initializer,shape='belt_e')
            
            user_input_s = tf.nn.embedding_lookup(embed_u_s,self.user_s)
            item_input_s = tf.nn.embedding_lookup(embed_i_s,self.item_s)
            user_input_t = tf.nn.embedding_lookup(embed_u_t,self.user_t)
            item_input_t = tf.nn.embedding_lookup(embed_i_t,self.item_t)
            user_input_s_t = tf.nn.embedding_lookup(embed_u_t,self.user_s)
            user_input_t_s = tf.nn.embedding_lookup(embed_u_s,self.user_t)
            alpha = tf.nn.embedding_lookup(alpha_e,self.user_s)
            belt = tf.nn.embedding_lookup(belt_e,self.user_t)
            
            user_input_s = alpha * user_input_s + (1-alpha)* user_input_s_t
            user_input_t = belt * user_input_t + (1-belt)* user_input_t_s
            gmf_v_s = tf.multiply(user_input_s,item_input_s)
            gmf_v_t = tf.multiply(user_input_t,item_input_t)
        with tf.name_scope('predict'):
            w_out_s = init_variable([self.GMFlayer,1],'w_out_s')
            b_out_s = init_variable([1],'b_out_s')
            self.y_s = tf.add(tf.matmul(gmf_v_s ,w_out_s),b_out_s)
            self.y_s = tf.reshape(self.y_s,[-1])

            w_out_t = init_variable([self.GMFlayer,1],'w_out_t')
            b_out_t = init_variable([1],'b_out_t')
            self.y_t = tf.add(tf.matmul(gmf_v_t ,w_out_t),b_out_t)
            self.y_t = tf.reshape(self.y_t,[-1])
    def add_loss(self):
        def loss_op(rate,y_):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rate ,logits=self.y_))
        self.loss = loss_op(self.rate_s,self.y_s)+ loss_op(self.rate_t,self.y_t)
    def run(self):
        print('='*20+'model Info'+'='*20)
        print(self.modelName)
        best_hr = -1
        best_NDCG = -1
        best_epoch = -1
        print("Start Training!")
        hr_list_s=[]
        hr_list_t=[]
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            t1= time.time()
            self.run_epoch()
            t2= time.time()
            print('='*50)
            print("Start Evaluation!")
            self.evaluate(epoch)
            t3 = time.time()
            print("Epoch ", epoch, ' [',t2-t1,'] ',"HR: {}, NDCG: {}, [{}] ".format(self.hr, self.NDCG,t3-t2))
            hr_list.append(self.hr)
            # plt.scatter(epoch,self.hr,c=self.color)
            if self.hr > best_hr or self.NDCG > best_NDCG:
                best_hr = self.hr
                best_NDCG = self.NDCG
                best_epoch = epoch
                self.saver.save(self.sess, self.checkPoint)
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
        print("="*20+"Epoch ", epoch, "End"+"="*20)    
        print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
        print("Training complete!")
        plt.plot(hr_list,label=self.modelName)

    def run_epoch(self, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            _, tmp_loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(tmp_loss)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}

    def evaluate(self,epoch):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0


        hr =[]
        NDCG = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = self.sess.run(self.y_, feed_dict=feed_dict)

            item_score_dict = {}
            lenth = len(testItem[i])
            for j in range(lenth):
                j = (j+1) % lenth
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            ranklist = heapq.nlargest(self.topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        self.hr, self.NDCG = np.mean(hr), np.mean(NDCG)


