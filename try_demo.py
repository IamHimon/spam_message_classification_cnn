# import jieba
# import time
# import datetime
#
# # 加载停用词
# stopwd = [line.strip() for line in open("data/spam_message/哈工大停用词表.txt")]
# stopwd.append('\n')
# stopwd2 = set('\n')
#
# with open("data/spam_message/spam_message_train_80W.txt") as file:
#     train_data = list(file.readlines())
#
#
# message = train_data[0].split("	")[2]
# seg_list = jieba.cut(message)
#
# print(set(seg_list)-set(stopwd2))
#
# from gensim.models.word2vec import Word2Vec
# model = Word2Vec.load_word2vec_format('/home/himon/Jobs/nlps/word2vec/sougo_vectors.bin', binary=True)
# print(model)
#
# # 计算某个词的相关词列表
# y2 = model.most_similar("周杰伦", topn=20)  # 20个最相关的
# print(u"和周杰伦最相关的词有：\n")
# for item in y2:
#     print(item[0], item[1])
# print("--------\n")
#
# l1 = ['a', 'b', 'c', 'd', 'e', 'fr']
# l2 = ['a', 'c']
# l3 = [e for e in l1 if e not in l2]
# print(l3)
# print('\u3000')
#
# print(l1[:-3])
# print(l1[-3:]) # -2:倒数第二个

# import numpy as np
#
# l = np.random.randint(0, 10)
# print(l)
#
# a = np.array([1, 2, 3])
# print(a)
#
# b = np.arange(0, 800000, 1)
# print(b)
# print(b.shape)

# #或者让一共有m块，自动分（尽可能平均）
# #split the arr into N chunks
# def chunks(arr, m):
#     n = int(math.ceil(len(arr) / float(m)))
#     return [arr[i:i + n] for i in range(0, len(arr), n)]

#
# n = 10
# l1 = [a for a in range(100)]
# l2 = [l1[i:i+n] for i in range(0, len(l1), n)]  # list均分十份
# print(l2)
# str = '　　有图有    真相哦~~~　　'
# str = str.strip().replace(' ', '')
# print(str)
# print(chunks(l1,10))
import random
# print(random.randint(0, 9))
#
# for i in range(10):
#     print(i)

# str1 = "hello usa"
# str = "-";
# seq = ("a", "b", "c") # 字符串序列
# str = ['hello china']
# r = " ".join(str)
# l2 = [str1]
# l1 = []
# l1.append(str1)
#
# r2 = " ".join(l1)
# print(r2)
# print(l2)
# print(l1)
# from sklearn import cross_validation
# import numpy as np
#
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [4, 6], [5, 7], [2, 3], [3,4], [4,5],[9,8]])
# y = np.array([1, 2, 3, 4])
# print(len(X))
# k_fold = cross_validation.KFold(len(X), n_folds=5, shuffle=True)
# for train_indices, test_indices in k_fold:
#     print('Train: %s | test: %s' % (train_indices, test_indices))
#     print("train data set:\n", X[train_indices])
#     print("test data set:\n", X[test_indices])


import tensorflow as tf

#our NN's output
logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
#step1:do softmax
y=tf.nn.softmax(logits)
#true label
y_=tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])

hh = y_*tf.log(y)
#step2:do cross_entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#do cross_entropy just one step
cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, y_))#dont forget tf.reduce_sum()!!

with tf.Session() as sess:
    softmax=sess.run(y)
    c_e = sess.run(cross_entropy)
    c_e2 = sess.run(cross_entropy2)
    r = sess.run(hh)
    print("step1:softmax result=")
    print(softmax)
    print("hh:")
    print(r)
    print("step2:cross_entropy result=")
    print(c_e)
    print("Function(softmax_cross_entropy_with_logits) result=")
    print(c_e2)






















