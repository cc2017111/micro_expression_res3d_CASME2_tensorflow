import tensorflow as tf
import video_opts
import numpy as np
from video_make_batch import get_batch
import video_Resnet_18
MAX_STEP = 50001
LEARNING_RATE = 0.0001
REGULARITION_RATE = 0.0001


def main():
    args = video_opts.parse_opts()
    channel = 3 if args.color else 1

    video_batch, label_batch = get_batch(args.output, args.batch_size)
    video_val_batch, label_val_batch = get_batch(args.output, args.batch_size)

    model = video_Resnet_18.Resnet_18(args.nclasses, args.depth, args.img_size, channel)
    model.inference().pred_func().accuracy_func().loss_func().train_func()

    summary_op = tf.summary.merge_all()

    check_path = args.checkpoint_path
    model_path = args.save_path

    acc_b_all = loss_b_all = np.array([])
    train_summary_count = val_summary_count = 0

    with tf.Session() as sess:
        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(initop)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        train_writter = tf.summary.FileWriter(check_path + 'train\\', sess.graph, max_queue=3)
        valid_writter = tf.summary.FileWriter(check_path + 'validation\\', max_queue=3)
        saver = tf.train.Saver()
        epoch = num_step = 0
        try:
            while epoch < args.epoch:
                step_count = int(args.num_data/args.batch_size)
                for __ in range(step_count):
                    if coord.should_stop():
                        break
                    _video_batch, _label_batch = sess.run([video_batch, label_batch])
                    feed_dict_train = {model.x: _video_batch, model.y: _label_batch}
                    _, train_accuracy, train_loss = sess.run([model.train_op(),
                                                              model.accuracy(),
                                                              model.loss()],
                                                             feed_dict=feed_dict_train)
                    acc_b_all = np.append(acc_b_all, train_accuracy)
                    loss_b_all = np.append(loss_b_all, train_loss)
                    num_step += 1
                    if num_step % 50 == 0 and num_step > 0:
                        mean_acc = np.mean(acc_b_all)
                        mean_loss = np.mean(loss_b_all)
                        print('epoch{:d} Step {:d}, train loss = {:.2f}, train accuracy = {:.2f}'.format(epoch, num_step, mean_loss,
                                                                                                         mean_acc * 100.0))

                        tf.summary.scalar('Accuracy', mean_acc)
                        tf.summary.scalar('Loss', mean_loss)
                        summary = sess.run(summary_op, feed_dict=feed_dict_train)
                        train_writter.add_summary(summary, global_step=train_summary_count * 50)
                        acc_b_all = loss_b_all = np.array([])
                        train_summary_count += 1

                    if num_step % 300 == 0 and num_step > 0:
                        _video_val_batch, _label_val_batch = sess.run([video_val_batch, label_val_batch])
                        feed_dict_val = {model.x: _video_val_batch, model.y: _label_val_batch}
                        acc_val, loss_val = sess.run([model.accuracy(), model.loss()], feed_dict=feed_dict_val)
                        print('epoch{:d} Step {:d}, val loss = {:.2f}, val accuracy = {:.2f}'.format(epoch, num_step, loss_val,
                                                                                                     acc_val * 100.0))

                        tf.summary.scalar('Accuracy_val', acc_val)
                        tf.summary.scalar('Loss_val', loss_val)
                        summary = sess.run(summary_op, feed_dict=feed_dict_val)
                        valid_writter.add_summary(summary, global_step=val_summary_count * 300)
                        val_summary_count += 1
                    if num_step % 500 == 0 and num_step > 0:
                        saver.save(sess, model_path, global_step=num_step)
                epoch += 1

        except tf.errors.OutOfRangeError:
            print("Done training -- epoch limit reached")
        finally:
            valid_writter.close()
            train_writter.close()
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    main()
