# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import math
import os
from tqdm import tqdm

class word2vec(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config['batch_size']
        self.embedding_size = config['embedding_size']
        self.num_sampled = config['num_sampled']
        self.filtered_vocaulary_size = config['filtered_vocaulary_size']
        
    def build_model(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        self.multi_articles = tf.placeholder(tf.int32, shape=[None, None])
        self.learning_rate = tf.placeholder(tf.float32, shape=[]) # Learning rate 는 linearly decay 됨

        embeddings = tf.Variable(tf.random_uniform([self.filtered_vocaulary_size, self.embedding_size], -0.05, 0.05))
        embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm

        # 아래: Similarity 구할 때 쓰는 변수
        # 대상 article 이 여러개 들어올 때 zero-padding index 로 사용할 용도로, embedding 뒤에 zero embedding 을 붙여줌 
        # ( CPU 에서 embedding_lookup 이 out-of-index 이면 error 가 남)
        zero_concat_embeddings = tf.concat([normalized_embeddings, tf.zeros([1, self.embedding_size])], axis=0)
        self.multi_embeddings = tf.nn.embedding_lookup(zero_concat_embeddings, self.multi_articles)
        
        self.elements_less_than_value = tf.less(self.multi_articles, self.filtered_vocaulary_size)
        as_ints = tf.cast(self.elements_less_than_value, tf.int32)
        seq_len = tf.cast(tf.reduce_sum(as_ints, 1), tf.float32)
        self.avg_embeddings = tf.math.divide(tf.reduce_sum(self.multi_embeddings, 1), tf.expand_dims(seq_len, -1))
        self.avg_embeddings = tf.reshape(self.avg_embeddings, [-1, self.embedding_size])

        self.similarity = tf.matmul(
            self.avg_embeddings, normalized_embeddings, transpose_b=True)

        # NCE_LOSS
        nce_weights = tf.Variable(
                tf.truncated_normal([self.filtered_vocaulary_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.filtered_vocaulary_size]))

        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=self.train_labels,
                inputs=embed,
                num_sampled=self.num_sampled,
                num_classes=self.filtered_vocaulary_size))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
        self.saver = tf.train.Saver(max_to_keep=3)


    def train(self, batch_inputs, batch_labels, alpha):
        feed = {self.train_inputs: batch_inputs, 
                self.train_labels: batch_labels,
                self.learning_rate: alpha}

        _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed)
        return loss_val
    
    def initialize_variables(self):
        self.sess.run(tf.global_variables_initializer())
        print("Variables initialized.")
    
    def store_checkpoint(self, ckpt_path="./ckpt/", step=0):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        
        self.saver.save(self.sess, os.path.join(ckpt_path, 'model.ckpt'), global_step=step)
        
    def restore_from_checkpoint(self, ckpt_path="./ckpt/", step=0, use_latest=True):
        latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        if use_latest:
            target_ckpt = latest_ckpt
            print ("Latest ckpt: {}".format(latest_ckpt))            
        else:
            target_ckpt = "{}model.ckpt-{}".format(ckpt_path, step)
            print ("Target ckpt: {}".format(target_ckpt))

        self.saver.restore(self.sess, target_ckpt)
        
    def most_similar(self, positives, idx2article_map, top_n=10):
        feed = {self.multi_articles: positives}
        print ("Calculate similarities...")
                        
        multi_embs, avg_embs, bests  = self.sess.run([self.multi_embeddings, self.avg_embeddings, self.similarity], feed_dict=feed)
        
        all_len = len(positives[0].flatten()) + top_n
        sims = (-bests).argsort()[:, :all_len]
        result = []
        for i, sim in tqdm(enumerate(sims)):
            tmp = [(idx2article_map[sim[n]], bests[i][sim[n]]) for n in range(all_len) if sim[n] not in positives[i]][:top_n]
            result.append(tmp)
        
        return multi_embs, avg_embs, result
    
