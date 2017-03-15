import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/home/himon/PycharmProjects/spam_message_classification_cnn/runs/1478012754/checkpoints",
                       "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_labels()
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["以上 比赛 规则 由 江苏 科技 大学 教职工 摄影 协会 负责 解释",
             "今天 有空 吗,我们 去 看 电影 吧,我 在 红楼 等 你",
             "红 都 百货 x 楼 婷美 专柜 x . x 节 活动 火热 进行 中 。 一年 仅 一次 的 最大 活动 力度 ！ 充值 送 ： 充 xxx 送 xxxxxxx 送 xxxxxxx 送 xxxxxxx 送 xxxxxxxx 送 xxxx 时间 ： x . xx - x . x 日 。 欢迎 各位 美女 们 前来 选购 ",
             "一次 价值 xxx 元 王牌 项目 ； 可 充值 xxx 元店 内 项目 卡 一张 ； 可以 参与 V 动好 生活 百分百 抽奖 机会 一次 ！ 预约 电话 ： ",
             "现有世纪丽都，三榆对面，xxx平米，超低团购房，xx楼采光好，单价xxxx，有意者可与我联系xxxxxxxxxxx孙",
             "三重好礼：新增定期x万以上可获精美礼品一份。xxx积分即可换xxx元购物卡或加油卡，多存多送，送完即止，请马上行动！活动截止至x.",
             "Twitter上已经转疯了的一张照片"]
    y_test = [1, 1, 0, 0, 0, 0, 1]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print(x_test.shape)
print(x_test)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        loss = graph.get_operation_by_name("output/scores").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_loss = sess.run(loss, feed_dict={input_x: x_test_batch, dropout_keep_prob: 1.0})
            print("batch_loss: ", batch_loss)

            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            print("batch_predictions:", batch_predictions)
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            print("all_predictions:", all_predictions)
            print(all_predictions.shape)


# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
