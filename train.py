import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import random
from sklearn import cross_validation

# Parameters
# ==================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================
# load data,
# x_text:splited sentences, y:label
print("Prepare data.")
x_text, y = data_helpers.load_sm_data_labels2()
print("x_text length:", len(x_text), "label length:", len(y))
# print(y)

# build vocabulary
print("build vocabulary.")
max_document_length = max([len(x.split(" ")) for x in x_text])
print("max_document_length:", max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
print(vocab_processor)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print(x.shape)
for i in x:
    print(i)
# print(x)
# x:(10662,56),构建一个词汇表，每个sentence对应一个list，一共10662个sentences，每个list是
# 56维度（最长sentence，不够的用0 pad），list的填充的是句子中word所对应的index。

# Randomly shuffle data
# print("shuffle data")
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]  # 重新洗牌的x list
# y_shuffled = y[shuffle_indices]  # 重新洗牌的y list
# print(y_shuffled)

# Split train/test set
# TODO: This is very crude, should use cross-validation
# x_train, x_dev = x_shuffled[:-10000], x_shuffled[-10000:]
# y_train, y_dev = y_shuffled[:-10000], y_shuffled[-10000:]
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Split train/test set, use 5_fold cross_validation
k_fold = cross_validation.KFold(len(x), n_folds=5, shuffle=True)
for train_indices, test_indices in k_fold:
    # print('Train: %s | test: %s' % (train_indices, test_indices))
    x_train_fold = x[train_indices]
    x_dev_fold = x[test_indices]
    y_train_fold = y[train_indices]
    y_dev_fold = y[test_indices]
    print("Train/Dev split: {:d}/{:d}".format(len(y_train_fold), len(y_dev_fold)))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train_fold.shape[1],
                num_classes=y_train_fold.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss) # the first part of `minimize()`
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step) # the second part of `minimize()`

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # checkpoint directory. Tensorflow assumes this directory already exits so we need to create it.
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # 创建Saver来管理模型中的所有变量, add ops to save and restore all the variables.
            saver = tf.train.Saver(tf.all_variables())

            # Write vocabulary, save the vocabulary to disk.
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # initializing the variables
            sess.run(tf.initialize_all_variables())

            # define a function for a single training step
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                # feed_dict来给运算图提供数据
                # 通过给run()函数输入feed_dict参数，来启动运算过程。
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train_fold, y_train_fold)), FLAGS.batch_size, FLAGS.num_epochs)
            # 把验证组数据(10000条)分为若干组(100),每次evalution随机选择一组来进行验证.
            x_dev_batches = data_helpers.chunks(x_dev_fold, 100)
            y_dev_batches = data_helpers.chunks(y_dev_fold, 100)
            # print(x_dev_batches)
            # print(y_dev_batches)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step) #  returns the value of global_step
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    # 从验证组的十组中随机挑选一组送入dev_step中
                    rand = random.randint(0, 99)
                    x_dev_batch = x_dev_batches[rand]
                    y_dev_batch = y_dev_batches[rand]

                    dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


