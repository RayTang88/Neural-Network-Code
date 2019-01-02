import numpy as np

def _one_hot():


    z=np.zeros(shape=[4,10])

    for i in range(4):
        index = int('0 1 2 3 4 5 6 7 8 9'.split(' ')[i])
        z[i][index]+=1
    return z
    # print(z)
if __name__=='__main__':
    result=_one_hot()
    print(result)



import tensorflow as tf

a=tf.constant([[1,2],[3,4]])
b=tf.expand_dims(a,axis=1)#增加矩阵维度
c=tf.tile(b,[1,4,1])

with tf.Session()as sess:
    print(sess.run(a).shape)
    print(sess.run(b).shape)
    print(sess.run(c).shape)