import argparse

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import far_ho as far
import experiment_manager as em
import numpy as np
import inspect, os, time
#from hr_resnet import hr_res_net_tcml_v1_builder, hr_res_net_tcml_Omniglot_builder
from shutil import copyfile
from models import hr_res_net_tcml_v1_builder
from threading import Thread
import pickle

from tensorflow.python.platform import flags
from far_ho.examples.hyper_representation import omniglot_model

import seaborn
seaborn.set_style('whitegrid', {'figure.figsize': (30, 20)})

em.DATASET_FOLDER = 'datasets'

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', type=str, default="train", metavar='STRING',
                    help='mode, can be train or test')

# GPU options
parser.add_argument('-vg', '--visible-gpus', type=str, default="1", metavar='STRING',
                    help="gpus that tensorflow will see")

# Dataset/method options
parser.add_argument('-d', '--dataset', type=str, default='miniimagenet', metavar='STRING',
                    help='omniglot or miniimagenet.')
parser.add_argument('-nc', '--classes', type=int, default=5, metavar='NUMBER',
                    help='number of classes used in classification (c for  c-way classification).')
parser.add_argument('-etr', '--examples_train', type=int, default=1, metavar='NUMBER',
                    help='number of examples used for inner gradient update (k for k-shot learning).')
parser.add_argument('-etes', '--examples_test', type=int, default=15, metavar='NUMBER',
                    help='number of examples used for test sets')

# Training options
parser.add_argument('-s', '--seed', type=int, default=0, metavar='NUMBER',
                    help='seed for random number generators')
parser.add_argument('-mbs', '--meta_batch_size', type=int, default=2, metavar='NUMBER',
                    help='number of tasks sampled per meta-update')
parser.add_argument('-nmi', '--n_meta_iterations', type=int, default=50000, metavar='NUMBER',
                    help='number of metatraining iterations.')
parser.add_argument('-T', '--T', type=int, default=5, metavar='NUMBER',
                    help='number of inner updates during training.')
parser.add_argument('-xi', '--xavier', type=bool, default=False, metavar='BOOLEAN',
                    help='FFNN weights initializer')
parser.add_argument('-bn', '--batch-norm', type=bool, default=False, metavar='BOOLEAN',
                    help='Use batch normalization before classifier')
parser.add_argument('-mlr', '--meta-lr', type=float, default=0.3, metavar='NUMBER',
                    help='starting meta learning rate')
parser.add_argument('-mlrdr', '--meta-lr-decay-rate', type=float, default=1.e-5, metavar='NUMBER',
                    help='meta lr  inverse time decay rate')
parser.add_argument('-cv', '--clip-value', type=float, default=0., metavar='NUMBER',
                    help='meta gradient clip value (0. for no clipping)')
parser.add_argument('-lr', '--lr', type=float, default=0.4, metavar='NUMBER',
                    help='starting learning rate')
parser.add_argument('-lrl', '--learn-lr', type=bool, default=False, metavar='BOOLEAN',
                    help='True if learning rate is an hyperparameter')

# Logging, saving, and testing options
parser.add_argument('-log', '--log', type=bool, default=False, metavar='BOOLEAN',
                    help='if false, do not log summaries, for debugging code.')
parser.add_argument('-ld', '--logdir', type=str, default='logs/', metavar='STRING',
                    help='directory for summaries and checkpoints.')
parser.add_argument('-res', '--resume', type=bool, default=True, metavar='BOOLEAN',
                    help='resume training if there is a model available')
parser.add_argument('-pi', '--print-interval', type=int, default=1, metavar='NUMBER',
                    help='number of meta-train iterations before print')
parser.add_argument('-si', '--save_interval', type=int, default=1, metavar='NUMBER',
                    help='number of meta-train iterations before save')
parser.add_argument('-te', '--test_episodes', type=int, default=600, metavar='NUMBER',
                    help='number of episodes for testing')

# Testing options (put parser.mode = 'test')
parser.add_argument('-exd', '--exp-dir', type=str, default=None, metavar='STRING',
                    help='directory of the experiment model files')
parser.add_argument('-itt', '--iterations_to_test', type=str, default=[40000], metavar='STRING',
                    help='meta_iteration to test (model file must be in "exp_dir")')

args = parser.parse_args()

available_devices = ('/gpu:0', '/gpu:1')
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

exp_string = str(args.classes) + 'way_' + str(args.examples_train) + 'shot_' + str(args.meta_batch_size) + 'mbs' \
             + str(args.T) + 'T' + str(args.clip_value) + 'cv' + str(args.meta_lr) + 'mlr' + str(args.lr)\
             + str(args.learn_lr) + 'lr'

dataset_load_dict = {'omniglot': em.load.meta_omniglot, 'miniimagenet': em.load.meta_mini_imagenet}
model_dict = {'omniglot': omniglot_model, 'miniimagenet': hr_res_net_tcml_v1_builder()}


def batch_producer(metadataset, batch_queue, n_batches, batch_size, rand=0):
    while True:
        batch_queue.put([d for d in metadataset.generate(n_batches, batch_size, rand)])


def start_batch_makers(number_of_workers, metadataset, batch_queue, n_batches, batch_size, rand=0):
    for w in range(number_of_workers):
        worker = Thread(target=batch_producer, args=(metadataset, batch_queue, n_batches, batch_size, rand))
        worker.setDaemon(True)
        worker.start()


# Class for debugging purposes for multi-thread issues (used now because it resolves rand issues)
class BatchQueueMock:
    def __init__(self, metadataset, n_batches, batch_size, rand):
            self.metadataset = metadataset
            self.n_batches = n_batches
            self.batch_size = batch_size
            self.rand = rand

    def get(self):
        return [d for d in self.metadataset.generate(self.n_batches, self.batch_size, self.rand)]


def save_obj(file_path, obj):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path, 'rb') as handle:
        b = pickle.load(handle)
    return b


''' Useful Functions '''


def feed_dicts(dat_lst, exs):
    dat_lst = em.as_list(dat_lst)
    train_fd = em.utils.merge_dicts(
        *[{_ex.x: dat.train.data, _ex.y: dat.train.target}
          for _ex, dat in zip(exs, dat_lst)])
    valid_fd = em.utils.merge_dicts(
        *[{_ex.x: dat.test.data, _ex.y: dat.test.target}
          for _ex, dat in zip(exs, dat_lst)])

    return train_fd, valid_fd


def just_train_on_dataset(dat, exs, far_ho, sess, T):
    train_fd, valid_fd = feed_dicts(dat, exs)
    # print('train_feed:', train_fd)  # DEBUG
    sess.run(far_ho.hypergradient.initialization)
    tr_acc, v_acc = [], []
    for ex in exs:
        # ts = io_opt.minimize(ex.errors['training'], var_list=ex.model.var_list).ts
        # ts = tf.train.GradientDescentOptimizer(lr).minimize(ex.errors['training'], var_list=ex.model.var_list)
        [sess.run(ex.optimizers['ts'], feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]}) for _ in range(T)]
        tr_acc.append(sess.run(ex.scores['accuracy'], feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]}))
        v_acc.append(sess.run(ex.scores['accuracy'], feed_dict={ex.x: valid_fd[ex.x], ex.y: valid_fd[ex.y]}))
    return tr_acc, v_acc


def accuracy_on(batch_queue, exs, far_ho, sess, T):
    tr_acc, v_acc = [], []
    for d in batch_queue.get():
        result = just_train_on_dataset(d, exs, far_ho, sess, T)
        tr_acc.extend(result[0])
        v_acc.extend(result[1])
    return tr_acc, v_acc


def just_train_on_dataset_up_to_T(dat, exs, far_ho, sess, T):
    train_fd, valid_fd = feed_dicts(dat, exs)
    # print('train_feed:', train_fd)  # DEBUG
    sess.run(far_ho.hypergradient.initialization)
    tr_acc, v_acc = [[] for _ in range(T)], [[] for _ in range(T)]
    for ex in exs:
        # ts = io_opt.minimize(ex.errors['training'], var_list=ex.model.var_list).ts
        # ts = tf.train.GradientDescentOptimizer(lr).minimize(ex.errors['training'], var_list=ex.model.var_list)
        for t in range(T):
            sess.run(ex.optimizers['ts'], feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]})
            tr_acc[t].append(sess.run(ex.scores['accuracy'], feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]}))
            v_acc[t].append(sess.run(ex.scores['accuracy'], feed_dict={ex.x: valid_fd[ex.x], ex.y: valid_fd[ex.y]}))
    return tr_acc, v_acc


def accuracy_on_up_to_T(batch_queue, exs, far_ho, sess, T):
    tr_acc, v_acc = [[] for _ in range(T)], [[] for _ in range(T)]
    for d in batch_queue.get():
        result = just_train_on_dataset_up_to_T(d, exs, far_ho, sess, T)
        [tr_acc[T].extend(r) for T, r in enumerate(result[0])]
        [v_acc[T].extend(r) for T, r in enumerate(result[1])]

    return tr_acc, v_acc


def build(metasets, hyper_model_builder, learn_lr, lr0, MBS, mlr0, mlr_decay, batch_norm_before_classifier, weights_initializer,
          process_fn=None):
    exs = [em.SLExperiment(metasets) for _ in range(MBS)]

    hyper_repr_model = hyper_model_builder(exs[0].x, 'HyperRepr')

    if learn_lr:
        lr = far.get_hyperparameter('lr', lr0)
    else:
        lr = tf.constant(lr0, name='lr')

    gs = tf.get_variable('global_step', initializer=0, trainable=False)
    meta_lr = tf.train.inverse_time_decay(mlr0, gs, decay_steps=1., decay_rate=mlr_decay)

    io_opt = far.GradientDescentOptimizer(lr)
    oo_opt = tf.train.AdamOptimizer(meta_lr)
    far_ho = far.HyperOptimizer()

    for k, ex in enumerate(exs):
        # print(k)  # DEBUG
        with tf.device(available_devices[k % len(available_devices)]):
            repr_out = hyper_repr_model.for_input(ex.x).out

            other_train_vars = []
            if batch_norm_before_classifier:
                batch_mean, batch_var = tf.nn.moments(repr_out, [0])
                scale = tf.Variable(tf.ones_like(repr_out[0]))
                beta = tf.Variable(tf.zeros_like(repr_out[0]))
                other_train_vars.append(scale)
                other_train_vars.append(beta)
                repr_out = tf.nn.batch_normalization(repr_out, batch_mean, batch_var, beta, scale, 1e-3)

            ex.model = em.models.FeedForwardNet(repr_out, metasets.train.dim_target,
                                                output_weight_initializer=weights_initializer, name='Classifier_%s' % k)

            ex.errors['training'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ex.y,
                                                                                           logits=ex.model.out))
            ex.errors['validation'] = ex.errors['training']
            ex.scores['accuracy'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ex.y, 1), tf.argmax(ex.model.out, 1)),
                                                           tf.float32), name='accuracy')

            # simple training step used for testing (look
            ex.optimizers['ts'] = tf.train.GradientDescentOptimizer(lr).minimize(ex.errors['training'],
                                                                                 var_list=ex.model.var_list)

            optim_dict = far_ho.inner_problem(ex.errors['training'], io_opt,
                                              var_list=ex.model.var_list + other_train_vars)
            far_ho.outer_problem(ex.errors['validation'], optim_dict, oo_opt,
                                 hyper_list=tf.get_collection(far.GraphKeys.HYPERPARAMETERS), global_step=gs)

    far_ho.finalize(process_fn=process_fn)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=240)
    return exs, far_ho, saver


def meta_train(exp_dir, metasets, exs, far_ho, saver, sess, n_test_episodes, MBS, seed, resume, T,
               n_meta_iterations, print_interval, save_interval):
    # use workers to fill the batches queues (is it worth it?)

    result_path = os.path.join(exp_dir, 'results.pickle')

    tf.global_variables_initializer().run(session=sess)

    n_test_batches = n_test_episodes // MBS
    rand = em.get_rand_state(seed)

    results = {'train_train': {'mean': [], 'std': []}, 'train_test': {'mean': [], 'std': []},
               'test_test': {'mean': [], 'std': []}, 'valid_test': {'mean': [], 'std': []},
               'outer_losses': {'mean':[], 'std': []}, 'learning_rate': [], 'iterations': [],
               'episodes': [], 'time': []}

    start_time = time.time()

    resume_itr = 0
    if resume:
        model_file = tf.train.latest_checkpoint(exp_dir)
        if model_file:
            print("Restoring results from " + result_path)
            results = load_obj(result_path)
            start_time = results['time'][-1]

            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:]) + 1
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    ''' Meta-Train '''
    train_batches = BatchQueueMock(metasets.train, 1, MBS, rand)
    valid_batches = BatchQueueMock(metasets.validation, n_test_batches, MBS, rand)
    test_batches = BatchQueueMock(metasets.test, n_test_batches, MBS, rand)

    print('\nIteration quantities: train_train acc, train_test acc, valid_test, acc'
          ' test_test acc mean(std) over %d episodes' % n_test_episodes)
    with sess.as_default():
        inner_losses = []
        for meta_it in range(resume_itr, n_meta_iterations):
            tr_fd, v_fd = feed_dicts(train_batches.get()[0], exs)

            far_ho.run(T, tr_fd, v_fd)
            # inner_losses.append(far_ho.inner_losses)

            outer_losses = [sess.run(ex.errors['validation'], v_fd) for ex in exs]
            outer_losses_moments = (np.mean(outer_losses), np.std(outer_losses))
            results['outer_losses']['mean'].append(outer_losses_moments[0])
            results['outer_losses']['std'].append(outer_losses_moments[1])

            # print('inner_losses: ', inner_losses[-1])

            if meta_it % print_interval == 0 or meta_it == n_meta_iterations - 1:
                results['iterations'].append(meta_it)
                results['episodes'].append(meta_it * MBS)

                train_result = accuracy_on(train_batches, exs, far_ho, sess, T)
                test_result = accuracy_on(test_batches, exs, far_ho, sess, T)
                valid_result = accuracy_on(valid_batches, exs, far_ho, sess, T)

                train_train = (np.mean(train_result[0]), np.std(train_result[0]))
                train_test = (np.mean(train_result[1]), np.std(train_result[1]))
                valid_test = (np.mean(valid_result[1]), np.std(valid_result[1]))
                test_test = (np.mean(test_result[1]), np.std(test_result[1]))

                duration = time.time() - start_time
                results['time'].append(duration)

                results['train_train']['mean'].append(train_train[0])
                results['train_test']['mean'].append(train_test[0])
                results['valid_test']['mean'].append(valid_test[0])
                results['test_test']['mean'].append(test_test[0])

                results['train_train']['std'].append(train_train[1])
                results['train_test']['std'].append(train_test[1])
                results['valid_test']['std'].append(valid_test[1])
                results['test_test']['std'].append(test_test[1])

                results['inner_losses'] = inner_losses

                print('mean outer losses: {}'.format(outer_losses_moments[1]))

                print('it %d, ep %d (%.2fs): %.3f, %.3f, %.3f, %.3f' % (meta_it, meta_it * MBS, duration, train_train[0],
                                                                      train_test[0], valid_test[0], test_test[0]))

                lr = sess.run(["lr:0"])[0]
                print('lr: {}'.format(lr))

                # do_plot(logdir, results)

            if meta_it % save_interval == 0 or meta_it == n_meta_iterations - 1:
                saver.save(sess, exp_dir + '/model' + str(meta_it))
                save_obj(result_path, results)

        return results


def meta_test(exp_dir, metasets, exs, far_ho, saver, sess, c_way, k_shot, lr,  n_test_episodes, MBS, seed, T,
              iterations=list(range(10000))):

    meta_test_str = str(c_way) + 'way_' + str(k_shot) + 'shot_' \
                 + str(T) + 'T' + str(lr) + 'lr' + str(n_test_episodes) + 'ep'

    n_test_batches = n_test_episodes // MBS
    rand = em.get_rand_state(seed)

    valid_batches = BatchQueueMock(metasets.validation, n_test_batches, MBS, rand)
    test_batches = BatchQueueMock(metasets.test, n_test_batches, MBS, rand)

    print('\nMeta-testing {} (over {} eps)...'.format(meta_test_str, n_test_episodes))

    test_results = {'test_test': {'mean': [], 'std': []}, 'valid_test': {'mean': [], 'std': []},
                    'cp_numbers': [], 'time': [],
                    'n_test_episodes': n_test_episodes, 'episodes': [], 'iterations': []}

    test_result_path = os.path.join(exp_dir, meta_test_str + '_results.pickle')

    start_time = time.time()
    for i in iterations:
        model_file = os.path.join(exp_dir, 'model' + str(i))
        if tf.train.checkpoint_exists(model_file):
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

            test_results['iterations'].append(i)
            test_results['episodes'].append(i * MBS)

            valid_result = accuracy_on(valid_batches, exs, far_ho, sess, T)
            test_result = accuracy_on(test_batches, exs, far_ho, sess, T)

            duration = time.time() - start_time

            valid_test = (np.mean(valid_result[1]), np.std(valid_result[1]))
            test_test = (np.mean(test_result[1]), np.std(test_result[1]))

            test_results['time'].append(duration)

            test_results['valid_test']['mean'].append(valid_test[0])
            test_results['test_test']['mean'].append(test_test[0])

            test_results['valid_test']['std'].append(valid_test[1])
            test_results['test_test']['std'].append(test_test[1])

            print('valid-test_test acc (%d meta_it)(%.2fs): %.3f (%.3f),  %.3f (%.3f)' % (i, duration, valid_test[0],
                                                                                          valid_test[1],test_test[0],
                                                                                          test_test[1]))

            save_obj(test_result_path, test_results)

    return test_results


def meta_test_up_to_T(exp_dir, metasets, exs, far_ho, saver, sess, c_way, k_shot, lr, n_test_episodes, MBS, seed, T,
              iterations=list(range(10000))):
    meta_test_str = str(c_way) + 'way_' + str(k_shot) + 'shot_' + str(lr) + 'lr' + str(n_test_episodes) + 'ep'

    n_test_batches = n_test_episodes // MBS
    rand = em.get_rand_state(seed)

    valid_batches = BatchQueueMock(metasets.validation, n_test_batches, MBS, rand)
    test_batches = BatchQueueMock(metasets.test, n_test_batches, MBS, rand)
    train_batches = BatchQueueMock(metasets.train, n_test_batches, MBS, rand)

    print('\nMeta-testing {} (over {} eps)...'.format(meta_test_str, n_test_episodes))

    test_results = {'valid_test': [], 'test_test': [], 'train_test': [], 'time': [], 'n_test_episodes': n_test_episodes,
                    'episodes': [], 'iterations': []}

    test_result_path = os.path.join(exp_dir, meta_test_str + 'noTrain_results.pickle')

    start_time = time.time()
    for i in iterations:
        model_file = os.path.join(exp_dir, 'model' + str(i))
        if tf.train.checkpoint_exists(model_file):
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

            test_results['iterations'].append(i)
            test_results['episodes'].append(i * MBS)

            valid_result = accuracy_on_up_to_T(valid_batches, exs, far_ho, sess, T)
            test_result = accuracy_on_up_to_T(test_batches, exs, far_ho, sess, T)
            train_result = accuracy_on_up_to_T(train_batches, exs, far_ho, sess, T)

            duration = time.time() - start_time

            test_results['time'].append(duration)

            for t in range(T):

                valid_test = (np.mean(valid_result[1][t]), np.std(valid_result[1][t]))
                test_test = (np.mean(test_result[1][t]), np.std(test_result[1][t]))
                train_test = (np.mean(train_result[1][t]), np.std(train_result[1][t]))

                if t >= len(test_results['valid_test']):
                    test_results['valid_test'].append({'mean': [], 'std': []})
                    test_results['test_test'].append({'mean': [], 'std': []})
                    test_results['train_test'].append({'mean': [], 'std': []})

                test_results['valid_test'][t]['mean'].append(valid_test[0])
                test_results['test_test'][t]['mean'].append(test_test[0])
                test_results['train_test'][t]['mean'].append(train_test[0])

                test_results['valid_test'][t]['std'].append(valid_test[1])
                test_results['test_test'][t]['std'].append(test_test[1])
                test_results['train_test'][t]['std'].append(train_test[1])

                print('valid-test_test acc T=%d (%d meta_it)(%.2fs): %.4f (%.4f), %.4f (%.4f),'
                      '  %.4f (%.4f)' % (t+1, i, duration, train_test[0], train_test[1], valid_test[0], valid_test[1],
                                         test_test[0], test_test[1]))

                #print('valid-test_test acc T=%d (%d meta_it)(%.2fs): %.4f (%.4f),'
                #      '  %.4f (%.4f)' % (t+1, i, duration, valid_test[0], valid_test[1],
                #                         test_test[0], test_test[1]))

            save_obj(test_result_path, test_results)

    return test_results


# training and testing function
def train_and_test(metasets, name_of_exp, hyper_model_builder, logdir='logs/', seed=None, lr0=0.04, learn_lr=False, mlr0=0.001,
          mlr_decay=1.e-5, T=5, resume=True, MBS=4, n_meta_iterations=5000, weights_initializer=tf.zeros_initializer,
          batch_norm_before_classifier=False, process_fn=None, save_interval=5000, print_interval=5000,
          n_test_episodes=1000):

    params = locals()
    print('params: {}'.format(params))

    ''' Problem Setup '''
    np.random.seed(seed)
    tf.set_random_seed(seed)

    exp_dir = logdir + '/' + name_of_exp
    print('\nExperiment directory:', exp_dir + '...')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    executing_file_path = inspect.getfile(inspect.currentframe())
    print('copying {} into {}'.format(executing_file_path, exp_dir))
    copyfile(executing_file_path, os.path.join(exp_dir, executing_file_path.split('/')[-1]))

    exs, far_ho, saver = build(metasets, hyper_model_builder,learn_lr, lr0, MBS, mlr0, mlr_decay,
                               batch_norm_before_classifier, weights_initializer, process_fn)

    sess = tf.Session(config=em.utils.GPU_CONFIG())

    meta_train(exp_dir, metasets, exs, far_ho, saver, sess, n_test_episodes, MBS, seed, resume, T,
                   n_meta_iterations, print_interval, save_interval)

    meta_test(exp_dir, metasets, exs, far_ho, saver, sess, args.classes, args.examples_train, lr0,
              n_test_episodes, MBS, seed, T, list(range(n_meta_iterations)))


# training and testing function
def build_and_test(metasets, exp_dir, hyper_model_builder, seed=None, lr0=0.04, T=5, MBS=4,
                   weights_initializer=tf.zeros_initializer, batch_norm_before_classifier=False,
                   process_fn=None, n_test_episodes=600, iterations_to_test=list(range(100000))):

    params = locals()
    print('params: {}'.format(params))

    mlr_decay = 1.e-5
    mlr0 = 0.001
    learn_lr = False

    ''' Problem Setup '''
    np.random.seed(seed)
    tf.set_random_seed(seed)

    exs, far_ho, saver = build(metasets, hyper_model_builder,learn_lr, lr0, MBS, mlr0, mlr_decay,
                               batch_norm_before_classifier, weights_initializer, process_fn)

    sess = tf.Session(config=em.utils.GPU_CONFIG())

    meta_test_up_to_T(exp_dir, metasets, exs, far_ho, saver, sess, args.classes, args.examples_train, lr0,
                      n_test_episodes, MBS, seed, T, iterations_to_test)


def main():
    print(args.__dict__)

    try:
        metasets = dataset_load_dict[args.dataset](
            std_num_classes=args.classes, std_num_examples=(args.examples_train*args.classes,
                                                             args.examples_test*args.classes))
    except KeyError:
        raise ValueError('dataset FLAG must be omniglot or miniimagenet')

    weights_initializer = tf.contrib.layers.xavier_initializer() if args.xavier else tf.zeros_initializer

    if args.clip_value > 0.:
        def process_fn(t):
            return tf.clip_by_value(t, -args.clip_value, args.clip_value)
    else:
        process_fn = None

    logdir = args.logdir + args.dataset

    hyper_model_builder = model_dict[args.dataset]

    if args.mode == 'train':
        train_and_test(metasets, exp_string, hyper_model_builder, logdir, seed=args.seed,
                       lr0=args.lr,
                       learn_lr=args.learn_lr, mlr0=args.meta_lr, mlr_decay=args.meta_lr_decay_rate, T=args.T,
                       resume=args.resume, MBS=args.meta_batch_size, n_meta_iterations=args.n_meta_iterations,
                       weights_initializer=weights_initializer, batch_norm_before_classifier=args.batch_norm,
                       process_fn=process_fn, save_interval=args.save_interval, print_interval=args.print_interval,
                       n_test_episodes=args.test_episodes)

    elif args.mode == 'test':
        build_and_test(metasets, args.exp_dir, hyper_model_builder, seed=args.seed, lr0=args.lr,
                       T=args.T, MBS=args.meta_batch_size, weights_initializer=weights_initializer,
                       batch_norm_before_classifier=args.batch_norm, process_fn=process_fn,
                       n_test_episodes=args.test_episodes, iterations_to_test=args.iterations_to_test)


if __name__ == "__main__":
    main()
