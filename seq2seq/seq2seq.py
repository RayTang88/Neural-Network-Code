# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image, ImageFont, ImageDraw

import traceback

batch_size = 100
width = 120
height = 60
channel = 3
train_path = r'/home/ray/datasets/identify/train'
test_path = r'/home/ray/datasets/identify/test'

class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, height, width, channel])
        self.y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 4, 10])
        self.encode = Encode()
        self.decode = Decode()

    def forward(self):
        self.net_x = self.encode.forward(self.x)
        self.output , self.test_output = self.decode.forward(self.net_x)

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
    def precision(self):
        self.test_y = tf.argmax(self.y[0], axis=1)
        self.test_out = tf.argmax(self.test_output[0], axis=1)
        self.acc = tf.reduce_mean(tf.cast(self.test_y == self.test_out, dtype=tf.float32))



class Encode:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[height*channel, 128], stddev=tf.sqrt(2/128), dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros([128], dtype=tf.float32))

    def forward(self, x):
        self.en_x1 = tf.transpose(x, [0, 2, 1, 3])
        self.en_x2 = tf.reshape(self.en_x1, shape=[batch_size*width, height*channel])
        self.en_x3 = tf.nn.relu(tf.matmul(self.en_x2, self.w1)+self.b1)
        self.en_x4 = tf.reshape(self.en_x3, shape=[batch_size, 120, 128])
        with tf.variable_scope('encode') as scope:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(128)
            init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            encode_output, encode_final_state = tf.nn.dynamic_rnn(lstm_cell, self.en_x4, initial_state=init_state,
                                                                  dtype=tf.float32, time_major=False, scope=scope)
            self.en_xo = tf.transpose(encode_output, [1, 0, 2])[-1]


            return self.en_xo



class Decode:
    def __init__(self):
        self.w2 = tf.Variable(tf.truncated_normal([128, 10]))
        self.b2 = tf.Variable(tf.zeros([10], dtype=tf.float32))

    def forward(self, y):
        self.de_y1 = tf.expand_dims(y, axis=1)#(100,128)-->(100,1,128)
        self.de_y2 = tf.tile(self.de_y1, [1, 4, 1])#(100,4,128)
        with tf.variable_scope('decode') as scope:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(128)
            init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            decode_output, decode_final_state = tf.nn.dynamic_rnn(lstm_cell, self.de_y2, initial_state=init_state,
                                                                  dtype=tf.float32, time_major=False, scope=scope)
            self.de_y3 = tf.reshape(decode_output, [batch_size * 4, 128])
            self.de_y4 = tf.matmul(self.de_y3, self.w2) + self.b2
            self.de_yo = tf.reshape(self.de_y4, shape=[batch_size, 4, 10])  # 还原成one_hot
            self.test_de_yo = tf.reshape(tf.nn.softmax(self.de_y4), shape=[batch_size, 4, 10])


        return self.de_yo ,self.test_de_yo

class Sample:
    def __init__(self, path):
        # try:
            self.datasets = []
            for filename in os.listdir(path):

                x = imgplt.imread(os.path.join(path, filename))/255-0.5
                y = filename.split('.')[0]
                y = self._one_hot(y)

                self.datasets.append([x, y])


        # except Exception as e:
        #     traceback.print_exc()



    def Get_batch(self, size):

        xs = []
        ys = []
        for _ in range(size):
            index = np.random.randint(0, len(self.datasets))
            xs.append(self.datasets[index][0])
            ys.append(self.datasets[index][1])

        return xs,ys
    def _one_hot(self,x):
        z = np.zeros(shape=[4, 10])

        for i in range(4):
            index = int(x[i])
            z[i][index] += 1

        return z


if __name__ == '__main__':
    start_time = time.time()
    net = Net()
    net.forward()
    net.backward()
    net.precision()
    init = tf.global_variables_initializer()
    font = ImageFont.truetype(font='arial.ttf', size=18)
    saver = tf.train.Saver()

    plt.ion()
    ax = []
    ay = []
    by = []
    with tf.Session()as sess:
        if os.path.exists('seq2seq_V0/'):
            saver.restore(sess, 'seq2seq_V0/seq2seq.ckpt')
        sess.run(init)
        for epoch in range(1500):
            xs, ys = Sample(train_path).Get_batch(batch_size)
            _, loss, _ = sess.run([net.optimizer, net.loss, net.output], feed_dict={net.x: xs, net.y: ys})


            if epoch % 100 == 0:
                test_xs, test_ys = Sample(test_path).Get_batch(batch_size)
                acc, test_out = sess.run([net.acc, net.test_out], feed_dict={net.x: test_xs, net.y: test_ys})
                # test_y = np.argmax(test_ys[0], axis=1)
                # test_out = np.argmax(test_output[0], axis=1)
                # acc = np.mean(np.array(test_y == test_out, dtype=np.float32))

                # plt.clf()
                plt.figure("acc&loss")
                ax.append(epoch)
                ay.append(loss)
                by.append(acc)
                plt.xlabel('epoch')
                plt.ylabel('acc&loss')
                plt.plot(ax, ay, 'ro-', label='loss')
                plt.plot(ax, by, 'b^-', label='acc')
                plt.legend(['loss', 'acc'])
                plt.show()
                plt.pause(0.1)

                plt.figure("number")
                image = Image.fromarray(np.uint8((test_xs[0]+0.5)*255))
                draw = ImageDraw.Draw(image)
                draw.text((0, 0), text=''.join(str(test_out)), font=font)
                plt.imshow(image)
                plt.show()
                plt.pause(0.1)

        use_time = time.time()-start_time
        print('总共耗时：', use_time)
        print('单次耗时：', use_time/epoch)
        plt.ioff()
        if not os.path.exists('seq2seq_V0/'):
            os.mkdir('seq2seq_V0/')
        saver.save(sess, 'seq2seq_V0/seq2seq.ckpt')


















