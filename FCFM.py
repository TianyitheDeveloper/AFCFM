import os
import numpy as np
import tensorflow as tf
from time import time
import argparse
import LoadData as DATA
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description="Run FCFM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-10m',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. '
                             '1: initialize from pretrain; '
                             '0: randomly initialize; '
                             '-1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=32,
                        help='Number of hidden factors.')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--net_channel', nargs='?', default='[32,32,32,32,32]',
                        help='net_channel, should be 6 layers here')
    parser.add_argument('--regs', nargs='?', default=[0,0,0],
                        help='Regularization for user and item embeddings, fully-connected weights, CNN filter weights.')

    return parser.parse_args()


class FCFM():
    def __init__(self,
                 user_field_M,
                 item_field_M,
                 pretrain_flag,
                 save_file,
                 hidden_factor,
                 epoch,
                 batch_size,
                 learning_rate,
                 keep,
                 regs,
                 random_seed=2020):
        self.user_field_M = user_field_M
        self.item_field_M = item_field_M
        self.pretrain_flag = pretrain_flag
        self.save_file = save_file
        self.hidden_factor = hidden_factor
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep = keep
        self.random_seed = random_seed
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.lambda_weight = regs[2]

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # create placeholder
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.positive_features = tf.placeholder(tf.int32, shape=[None, None])
            self.negative_features = tf.placeholder(tf.int32, shape=[None, None])
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            # create variables
            self.weights = self._initialize_weights()
            self.nc = eval(args.net_channel)
            iszs = [1] + self.nc[:-1]
            oszs = self.nc
            self.P = []
            self.P.append(self._conv_weight(iszs[0], oszs[0]))
            for i in range(1, len(self.nc)):
                self.P.append(self._conv_weight(iszs[i], oszs[i]))
            self.W = self.weight_variable([self.nc[-1], 1])
            self.b = self.weight_variable([1])

            # FCFM (positive)
            self.user_feature_embeddings = tf.nn.embedding_lookup(
                self.weights['user_feature_embeddings'],self.user_features)
            self.positive_item_embeddings = tf.nn.embedding_lookup(
                self.weights['item_feature_embeddings'],self.positive_features)
            self.user_embedding = tf.reduce_sum(self.user_feature_embeddings, 1, keep_dims=True)
            self.positive_embedding = tf.reduce_sum(self.positive_item_embeddings, 1,keep_dims=True)
            self.interaction_image_positive = tf.matmul(tf.transpose(self.user_embedding, perm=[0, 2, 1]),
                                                        self.positive_embedding)
            self.net_input = tf.expand_dims(self.interaction_image_positive, -1)
            self.layer = []
            positive_input = self.net_input
            for p in self.P:
                self.layer.append(
                    self._conv_layer(positive_input, p))
                positive_input = self.layer[-1]
            self.dropout_positive = tf.nn.dropout(self.layer[-1], self.dropout_keep)
            self.interaction_positive = tf.matmul(tf.reshape(self.dropout_positive, [-1, self.nc[-1]]), self.W) + self.b
            self.user_feature_bias = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features), 1)
            self.item_feature_bias_positive = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.positive_features), 1)
            self.positive = tf.add_n(
                [self.interaction_positive, self.user_feature_bias, self.item_feature_bias_positive])

            # FCFM (negative)
            self.negative_item_embeddings = tf.nn.embedding_lookup(
                self.weights['item_feature_embeddings'],self.negative_features)
            self.negative_embedding = tf.reduce_sum(self.negative_item_embeddings, 1, keep_dims=True)
            self.interaction_image_negative = tf.matmul(
                tf.transpose(self.user_embedding, perm=[0, 2, 1]), self.negative_embedding)
            self.net_input = tf.expand_dims(self.interaction_image_negative, -1)
            self.layer = []
            negative_input = self.net_input
            for p in self.P:
                self.layer.append(
                    self._conv_layer(negative_input, p,))
                negative_input = self.layer[-1]
            self.dropout_negative = tf.nn.dropout(self.layer[-1], self.dropout_keep)
            self.interaction_negative = tf.matmul(tf.reshape(self.dropout_negative, [-1, self.nc[-1]]), self.W) + self.b
            self.item_feature_bias_negative = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.negative_features),1)
            self.negative = tf.add_n(
                [self.interaction_negative, self.user_feature_bias, self.item_feature_bias_negative])

            # create BPR loss
            self.loss = -tf.log(tf.sigmoid(self.positive - self.negative))
            self.loss = tf.reduce_sum(self.loss)
            self.loss = self.loss \
                        + self.lambda_bilinear * (tf.reduce_sum(tf.square(self.user_embedding))
                                                  + tf.reduce_sum(tf.square(self.positive_embedding))
                                                  + tf.reduce_sum(tf.square(self.negative_embedding))) \
                        + self.gamma_bilinear * self._regular([(self.W, self.b)]) \
                        + self.lambda_weight * (self._regular(self.P)
                                                + self._regular([(self.W, self.b)]))

            # create_optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv_weight(self, isz, osz):
        return (self.weight_variable([2, 2, isz, osz]), self.bias_variable([osz]))

    def _conv_layer(self, input, P):
        conv = tf.nn.conv2d(input, P[0], strides=[1, 2, 2, 1], padding='VALID')
        return tf.nn.relu(conv + P[1])

    def _regular(self, params):
        res = 0
        for param in params:
            res += tf.reduce_sum(tf.square(param[0])) + tf.reduce_sum(tf.square(param[1]))
        return res

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_feature_embeddings = pretrain_graph.get_tensor_by_name('user_feature_embeddings:0')
            item_feature_embeddings = pretrain_graph.get_tensor_by_name('item_feature_embeddings:0')
            user_feature_bias = pretrain_graph.get_tensor_by_name('user_feature_bias:0')
            item_feature_bias = pretrain_graph.get_tensor_by_name('item_feature_bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie, ub, ib = sess.run(
                    [user_feature_embeddings, item_feature_embeddings, user_feature_bias, item_feature_bias])
            all_weights['user_feature_embeddings'] = tf.Variable(ue, dtype=tf.float32)
            all_weights['item_feature_embeddings'] = tf.Variable(ie, dtype=tf.float32)
            all_weights['user_feature_bias'] = tf.Variable(ub, dtype=tf.float32)
            all_weights['item_feature_bias'] = tf.Variable(ib, dtype=tf.float32)
            print("load!")
        else:
            all_weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.1),
                name='user_feature_embeddings')  # user_field_M * K
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.1),
                name='item_feature_embeddings')  # item_field_M * K
            all_weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.user_field_M, 1], 0.0, 0.1), name='user_feature_bias')
            all_weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.item_field_M, 1], 0.0, 0.1), name='item_feature_bias')
        return all_weights

    def partial_fit(self, data):
        feed_dict = {self.user_features: data['X_user'], self.positive_features: data['X_positive'],
                     self.negative_features: data['X_negative'], self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, train_data, batch_size):
        X_user, X_positive, X_negative = [], [], []
        all_items = data.binded_items.values()
        while len(X_user) < batch_size:
            index = np.random.randint(0, len(train_data['X_user']))
            X_user.append(train_data['X_user'][index])
            X_positive.append(train_data['X_item'][index])
            user_features = "-".join([str(item) for item in train_data['X_user'][index][0:]])
            user_id = data.binded_users[user_features]
            pos = data.user_positive_list[user_id]
            neg = np.random.randint(len(all_items))
            while neg in pos:
                neg = np.random.randint(len(all_items))
            negative_feature = data.item_map[neg].strip().split('-')
            X_negative.append([int(item) for item in negative_feature[0:]])
        return {'X_user': X_user, 'X_positive': X_positive, 'X_negative': X_negative}

    def train(self, Train_data):
        for epoch in range(self.epoch):
            t1 = time()
            total_loss = 0
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            for i in range(total_batch):
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                loss = self.partial_fit(batch_xs)
                total_loss = total_loss + loss
            t2 = time()
            logger.info("the total loss in %d th iteration is: %f [%.2f s]" % (epoch, total_loss, t2 - t1))
            if (epoch + 1) in [1,5,10,20,50,100,200,300,400,500]:
                t3 = time()
                model.evaluate()
                t4 = time()
                print("evaluate: [%.2f s]" % (t4 - t3))
        print("end train begin save")
        if self.pretrain_flag < 0:
            print("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)

    def evaluate(self):
        self.graph.finalize()
        count = [0, 0, 0, 0]
        rank = [[], [], [], []]
        topK = [5, 10, 15, 20]
        for index in range(len(data.Test_data['X_user'])):
            user_features = data.Test_data['X_user'][index]
            item_features = data.Test_data['X_item'][index]
            scores = model.get_scores_per_user(user_features)
            true_item_id = data.binded_items["-".join([str(item) for item in item_features[0:]])]
            true_item_score = scores[true_item_id]
            user_id = data.binded_users["-".join([str(item) for item in user_features[0:]])]
            visited = data.user_positive_list[user_id]
            scores = np.delete(scores, visited)
            sorted_scores = sorted(scores, reverse=True)

            label = []
            for i in range(len(topK)):
                label.append(sorted_scores[topK[i] - 1])
                if true_item_score >= label[i]:
                    count[i] = count[i] + 1
                    rank[i].append(sorted_scores.index(true_item_score) + 1)
        for i in range(len(topK)):
            mrr = 0
            ndcg = 0
            hit_rate = float(count[i]) / len(data.Test_data['X_user'])
            for item in rank[i]:
                mrr = mrr + float(1.0) / item
                ndcg = ndcg + float(1.0) / np.log2(item + 1)
            mrr = mrr / len(data.Test_data['X_user'])
            ndcg = ndcg / len(data.Test_data['X_user'])
            k = (i + 1) * 5
            logger.info("top:%f" % k)
            logger.info("the Hit Rate is: %f" % hit_rate)
            logger.info("the MRR is: %f" % mrr)
            logger.info("the NDCG is: %f" % ndcg)

    def get_scores_per_user(self, user_features):
        X_user, X_item = [], []
        all_items = data.binded_items.values()
        for itemID in range(len(all_items)):
            X_user.append(user_features)
            item_feature = [int(feature) for feature in data.item_map[itemID].strip().split('-')[0:]]
            X_item.append(item_feature)
        feed_dict = {
            self.user_features: X_user,
            self.positive_features: X_item,
            self.train_phase: False,
            self.dropout_keep: 1.0
        }
        scores = self.sess.run((self.positive), feed_dict=feed_dict)
        scores = scores.reshape(len(all_items))
        return scores


if __name__ == '__main__':
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('FCFM.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset)
    save_file = 'pretrain-FCFM-%s/%s_%d' % (args.dataset, args.dataset, args.hidden_factor)
    t1 = time()
    model = FCFM(data.user_field_M,
                 data.item_field_M,
                 args.pretrain,
                 save_file,
                 args.hidden_factor,
                 args.epoch,
                 args.batch_size,
                 args.lr,
                 args.keep_prob,
                 args.regs)

    print("begin train")
    model.train(data.Train_data)
    print("end train")
    print("finish")
