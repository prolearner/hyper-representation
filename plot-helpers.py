import pickle
import os


def do_plot(exp_path, results=None):
    import pickle, os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if results is None:
        def load_obj(file_path):
            with open(file_path, 'rb') as handle:
                b = pickle.load(handle)
            return b
        results = load_obj(os.path.join(exp_path, 'results.pickle'))
        # results = load_obj(os.path.join(exp_path, 'test_results.pickle'))
        results = load_obj(os.path.join(exp_path, 'test5shot_results.pickle'))

    exp_name = exp_path.split('/')[-1]

    x_div = 1000
    x_string = 'episodes'  # can be 'iterations' or 'episodes'
    x_values = np.array(results[x_string]) / x_div

    arg_max_valid = np.argmax(results['valid_test']['mean'])
    chosen_valid = results['valid_test']['mean'][arg_max_valid]
    chosen_test = results['test_test']['mean'][arg_max_valid]
    chosen_valid_ci = results['valid_test']['std'][arg_max_valid]
    chosen_test_ci = results['test_test']['std'][arg_max_valid]
    chosen_x = x_values[arg_max_valid]

    fig = plt.figure(figsize=(8, 5))
    plt.title(exp_name)
    #plt.plot(x_values, results['train_train']['mean'], label='train_train')
    #plt.plot(x_values, results['train_test']['mean'], label='train_test')
    plt.plot(x_values, results['valid_test']['mean'], label='valid_test (t: %.4f (%.4f), x: %d)' % (chosen_valid,
                                                                                                    chosen_valid_ci,
                                                                                                    chosen_x))
    plt.plot(x_values, results['test_test']['mean'], label='test_test (t: %.4f (%.4f), x: %d)' % (chosen_test,
                                                                                                  chosen_test_ci,
                                                                                                  chosen_x))
    plt.legend(loc=0)
    plt.xlabel(x_string + ' / %d' % x_div)
    plt.savefig(exp_path + '/5shot_accuracies.png')
    plt.close(fig)


def load_results(exp_path):
    with open(os.path.join(exp_path, 'results.pickle'), 'rb') as handle:
        b = pickle.load(handle)
    return b