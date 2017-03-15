import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    uses an embedding layer,followed by a convolutional,max-poling and softmax layer.
    """
    # self 指的是类实例本身，
    # __init__: python定义类中的定义的构造器
    def __init__(self, sequence_length, num_classes, vocab_size
                 , embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        :param sequence_length: the length of out sentences(59 for our data set).
        :param num_classes:number of classes in the output layer,two in our case(pos,neg).
        :param vocab_size:the size of our vocabulary.
        :param embedding_size:the dimensionality of our embeddings.
        :param filter_sizes:the number of words we want our convolutional filters to cover([3,4,5] used in our case).
        :param num_filters:The number of filters per filter size.每个size的filter的filter的种类.
        :return:
        """

        # placeholders for input,output and dropout
        '''
        TensorFlow也提供这样的机制:先创建特定数据类型的占位符(placeholder)，之后再进行数据的填充.
        设计placeholder节点的唯一意图就是为了提供数据供给(feeding)的方法，
        placeholder节点被声明的时候是未被初始化的，也不包含数据，所以在train过程时，
        需要为这些placeholder节点提供数据.(所以train.py中通过feed_dict供给数据)
        '''
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # embedding layer, aim shape: [None, sequence_length, embedding_size, 1]
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # get a 3-D tensor:[None, sequence_length, embedding_size]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)  # expand into 4-D tensor

        # create a convolution + maxpool layer for each filter size
        pooled_output = []
        for i, filter_size in enumerate(filter_sizes):  # (0, seq[0]), (1, seq[1]), (2, seq[2]),
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # convolution layer
                # 前两维是patch大小,亦即卷积核大小:filter_size*embedding_size, 第三维是输入通道数(1个channel),第四维是输出通道的个数.
                # W: a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # 每个输出通道都要有一个偏置值,所以shape是[num_filters]
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # now get output of shape [1, sequence_length - filter_size + 1, 1, 1]

                # Max-pooling over the outputs
                # the size of the window: (sequence_length - filter_size + 1)*1,第二维大小:sequence_length - filter_size + 1,
                # The stride of the sliding window: 1,每步只动一个,因为窗口的大小刚好等于上层(convolution+nonlinearity)输出的维度,
                # 所以,max_pool的结果就是一维的,每个输出通道有一个值,所以第四维是num_filter,这第四维也就对应求得的feature.
                # max_pool的输出张量shape是[patch_size,height,width,out_channels]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool"
                )
                pooled_output.append(pooled)
        # output of shape [batch_size, 1, 1, num_filters]

        # combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # 第四维链接 ,long featere vector of shape [batch_size, num_filters_total]
        self.h_pool = tf.concat(3, pooled_output)
        # flatten the dimension,如: [1,2,3,4,5,6,,,,]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        #
        # # final(unnormalized) scores and predictions
        # with tf.name_scope("output"):
        #     W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
        #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")    # biases
        #     self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        #     self.predictions = tf.arg_max(self.scores, 1, name="predictions")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores") # Computes matmul(x, weights) + biases
            self.predictions = tf.argmax(self.scores, 1, name="predictions") # 第二维的最大值

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # calculate accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.arg_max(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")




