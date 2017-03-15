# -*- coding: utf-8 -*-：
import numpy as np
import re
import jieba
from collections import defaultdict
import pandas as pd
import sys
import _pickle as cPickle


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for orig_rev in f:
            # print("line:", line.strip().lower())
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 1,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0, cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for orig_rev in f:
            # print("line:", line.strip().lower())
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 0,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0, cv)}
            revs.append(datum)
    return revs, vocab


def get_w(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


# #arr是被分割的list，n是要分多少组。
def chunks(arr, n):
    patch = int(len(arr)/n)
    return [arr[i:i+patch] for i in range(0, len(arr), patch)]


# 分词并保存下来结果
def corpus_segment(corpus_path):
    print(corpus_path)
    f = open("output_seg.txt", "w")
    # 加载停用词
    stopwd = [line for line in open("data/stopwords/哈工大停用词表.txt")]
    # print("stopwd\n", stopwd)

    with open(corpus_path) as file:
        lines = file.readlines()
        for line in lines:
            str1 = ''
            split_list = line.split('\t')
            message = split_list[2].strip().replace(' ', '')     # 过滤点开头和结尾的空格,过滤掉中间的空格
            seg_list = jieba.lcut(message)
            if seg_list:
                seg_list.pop()
                # print(split_list[0], seg_list)
                # 过滤掉停用词
                seg_list = [word for word in seg_list if word not in stopwd]
                # 把list转化为str类型的,不同元素之间用空格分开
                for e in seg_list:
                    str1 += e+' '

                print(split_list[0]+'\t'+split_list[1]+'\t'+str1+'\n')
                # 写入文件
                f.write(split_list[0]+'\t'+split_list[1]+'\t'+str1+'\n')


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str2(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def load_data_labels():
    """
    loads MR polarity data from files,splits the data into words and generates labels.
    returns split sentences and labels.
    :return:list[[ps1,ps2,ps3,,,,,ns1,ns2,ns3,,,],[pl1,pl2,pl3,,,,,nl1,nl2,nl3,,,]]
            ps1：splited positive sentences list,[w1,w2,w3,,,]
            pl1:positive label([0,1])
            ns1：splited negative sentences list,[w1,w2,w3,,,]
            nl1:negative label([1,0])
    """
    # Load data from files, convert content per line to  list[string]
    positive_examples = list(open("/home/himon/Jobs/corpus/rt-polaritydata/rt-polarity.pos", "r", encoding="ISO-8859-1")
                             .readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("/home/himon/Jobs/corpus/rt-polaritydata/rt-polarity.neg", "r", encoding="ISO-8859-1")
                             .readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels, all element is [0,1] label positive case or [1,0] label negative case.
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)  # join a sequence of arrays
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有yield说明不是一个普通函数，是一个Generator.
    函数效果：对data，一共分成num_epochs个阶段（epoch），在每个epoch内，如果shuffle=True，就将data重新洗牌，
    批量生成(yield)一批一批的重洗过的data，每批大小是batch_size，一共生成int(len(data)/batch_size)+1批。
    Generate a  batch iterator for a dataset.
    :param data:
    :param batch_size:每批data的size
    :param num_epochs:阶段数目
    :param shuffle:洗牌
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batch_per_epoch = int(len(data)/batch_size) + 1  # 每段的batch数目
    for epoch in range(num_epochs):
        if shuffle:
            # np.random.permutation(),得到一个重新排列的序列(Array)
            # np.arrange(),得到一个均匀间隔的array.
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]    # 重新洗牌的data
        else:
            shuffle_data = data
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]   # all elements index between start_index and end_index


def load_sm_data_labels():
    with open("data/stanfordSegResult.txt") as file:
        examples = list(file.readlines())
        examples = [s.strip() for s in examples]
        raw_text = [list(line.split(", ")) for line in examples]
        # 过滤掉格式不对的数据
        x_text = [x for x in raw_text if len(x) == 2]

        sentence = [sent[1] for sent in x_text]
        raw_label = [sent[0] for sent in x_text]

        label = []
        for l in raw_label:
            if l == '0':   # positive
                label.append([0, 1])
            if l == '1':   # negative
                label.append([1, 0])
        label1 = label[:-400000]
        label2 = label[-400000:]
        y = np.concatenate([label1, label2], 0)

        return [sentence, y]


def load_sm_data_labels2():
    with open("test_output_seg.txt") as file:
        examples = list(file.readlines())
        examples = [s.strip() for s in examples]
        raw_text = [list(line.split('\t')) for line in examples]

        # 过滤掉格式不对的数据
        x_text = [x for x in raw_text if len(x) == 3]
        # print("text count:", len(x_text))
        sentence = [sent[2] for sent in x_text]

        # max_document_length = max([len(x.split(" ")) for x in sentence])
        # print("max_document_length:", max_document_length)
        # for a in sentence:
        #     if len(a.split(" ")) == max_document_length:
        #         print(a)
        #
        raw_label = [sent[1] for sent in x_text]
        # print("label count:", len(raw_label))

        label = []
        for l in raw_label:
            if l == '0':   # positive
                label.append([0, 1])
            if l == '1':   # negative
                label.append([1, 0])
        label1 = label[:-20]
        label2 = label[-20:]
        y = np.concatenate([label1, label2], 0)
        # print(y)
        # print("label list shape:", y.shape)

        return [sentence, y]

if __name__ == '__main__':
    print('main')
    w2v_file = "/home/himon/Jobs/nlps/word2vec/vectors.bin"
    # corpus_segment("data/spam_message/spam_message_train_80W.txt")
    # print(load_sm_data_labels2())
    # r = load_sm_data_labels2()
    # print(r)
    # data_folder = ["/home/himon/Jobs/corpus/rt-polaritydata/rt-polarity.pos",
    #                "/home/himon/Jobs/corpus/rt-polaritydata/rt-polarity.neg"]
    # print("loading data...")
    # revs, vocab = build_data_cv(data_folder, cv=10, clean_string=False)
    # max_l = np.max(pd.DataFrame(revs)["num_words"])
    # print("data loaded!")
    # print("number of sentences: " + str(len(revs)))
    # print("vocab size: " + str(len(vocab)))
    # print("max sentence length: " + str(max_l))
    # print("loading word2vec vectors...")
    # w2v = load_bin_vec(w2v_file, vocab)
    # print("word2vec loaded!")
    # print("num words already in word2vec: " + str(len(w2v)))
    # add_unknown_words(w2v, vocab)
    # W, word_idx_map = get_w(w2v)
    # rand_vecs = {}
    # add_unknown_words(rand_vecs, vocab)
    # W2, _ = get_w(rand_vecs)
    # cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    # print("dataset created!")
    x, y = load_sm_data_labels2()
    print(x)



















