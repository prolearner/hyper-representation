import experiment_manager as em
from experiment_manager import models
import tensorflow as tf
from tensorflow.contrib import layers as tcl
import far_ho as far


class TCML_ResNet(models.Network):
    def __init__(self, _input, name=None, deterministic_initialization=False, reuse=False):
        self.var_coll = far.HYPERPARAMETERS_COLLECTIONS
        super().__init__(_input, name, deterministic_initialization, reuse)


        self.betas = self.filter_vars('beta')
        self.moving_means = self.filter_vars('moving_mean')
        self.moving_variances = self.filter_vars('moving_variance')

        if not reuse:
            far.utils.remove_from_collection(far.GraphKeys.MODEL_VARIABLES, *self.moving_means, *self.moving_variances)

        far.utils.remove_from_collection(far.GraphKeys.HYPERPARAMETERS, *self.moving_means, *self.moving_variances)
        print(name, 'MODEL CREATED')

    def _build(self):

        def residual_block(x, n_filters):
            skip_c = tcl.conv2d(x, n_filters, 1, activation_fn=None)

            def conv_block(xx):
                out = tcl.conv2d(xx, n_filters, 3, activation_fn=None, normalizer_fn=tcl.batch_norm,
                                 variables_collections=self.var_coll)
                return em.utils.leaky_relu(out, 0.1)

            out = x
            for _ in range(3):
                out = conv_block(out)

            add = tf.add(skip_c, out)

            return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        self + residual_block(self.out, 64)
        self + residual_block(self.out, 96)
        self + residual_block(self.out, 128)
        self + residual_block(self.out, 256)
        self + tcl.conv2d(self.out, 2048, 1, variables_collections=self.var_coll)
        self + tf.nn.avg_pool(self.out, [1, 6, 6, 1], [1, 6, 6, 1], 'VALID')
        self + tcl.conv2d(self.out, 512, 1, variables_collections=self.var_coll)
        self + tf.reshape(self.out, (-1, 512))

    def for_input(self, new_input):
        return TCML_ResNet(new_input, self.name, self.deterministic_initialization, True)


class TCML_ResNet_Omniglot(models.Network):
    def __init__(self, _input, name=None, deterministic_initialization=False, reuse=False):
        self.var_coll = far.HYPERPARAMETERS_COLLECTIONS
        super().__init__(_input, name, deterministic_initialization, reuse)


        self.betas = self.filter_vars('beta')
        self.moving_means = self.filter_vars('moving_mean')
        self.moving_variances = self.filter_vars('moving_variance')

        if not reuse:
            far.utils.remove_from_collection(far.GraphKeys.MODEL_VARIABLES, *self.moving_means, *self.moving_variances)

        far.utils.remove_from_collection(far.GraphKeys.HYPERPARAMETERS, *self.moving_means, *self.moving_variances)
        print(name, 'MODEL CREATED')

    def _build(self):

        def residual_block(x, n_filters):
            skip_c = tcl.conv2d(x, n_filters, 1, activation_fn=None)

            def conv_block(xx):
                out = tcl.conv2d(xx, n_filters, 3, activation_fn=None, normalizer_fn=tcl.batch_norm,
                                 variables_collections=self.var_coll)
                return em.utils.leaky_relu(out, 0.1)

            out = x
            for _ in range(3):
                out = conv_block(out)

            add = tf.add(skip_c, out)

            return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        self + residual_block(self.out, 64)
        self + residual_block(self.out, 96)
        # self + residual_block(self.out, 128)
        # self + residual_block(self.out, 256)
        self + tcl.conv2d(self.out, 2048, 1, variables_collections=self.var_coll)
        self + tf.nn.avg_pool(self.out, [1, 6, 6, 1], [1, 6, 6, 1], 'VALID')
        self + tcl.conv2d(self.out, 512, 1, variables_collections=self.var_coll)
        self + tf.reshape(self.out, (-1, 512))

    def for_input(self, new_input):
        return TCML_ResNet_Omniglot(new_input, self.name, self.deterministic_initialization, True)


class TCML_ResNet_Omniglot_v2(models.Network):
    def __init__(self, _input, name=None, deterministic_initialization=False, reuse=False):
        self.var_coll = far.HYPERPARAMETERS_COLLECTIONS
        super().__init__(_input, name, deterministic_initialization, reuse)

        self.betas = self.filter_vars('beta')
        self.moving_means = self.filter_vars('moving_mean')
        self.moving_variances = self.filter_vars('moving_variance')

        if not reuse:
            far.utils.remove_from_collection(far.GraphKeys.MODEL_VARIABLES, *self.moving_means, *self.moving_variances)

        far.utils.remove_from_collection(far.GraphKeys.HYPERPARAMETERS, *self.moving_means, *self.moving_variances)
        print(name, 'MODEL CREATED')

    def _build(self):

        def residual_block(x, n_filters):
            skip_c = tcl.conv2d(x, n_filters, 1, activation_fn=None)

            def conv_block(xx):
                out = tcl.conv2d(xx, n_filters, 3, activation_fn=None, normalizer_fn=tcl.batch_norm,
                                 variables_collections=self.var_coll)
                return em.utils.leaky_relu(out, 0.1)

            out = x
            for _ in range(3):
                out = conv_block(out)

            add = tf.add(skip_c, out)

            return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        self + residual_block(self.out, 64)
        self + residual_block(self.out, 96)
        self + residual_block(self.out, 128)
        self + residual_block(self.out, 256)
        self + tcl.conv2d(self.out, 2048, 1, variables_collections=self.var_coll)
        self + tf.nn.avg_pool(self.out, [1, 6, 6, 1], [1, 6, 6, 1], 'SAME')
        self + tcl.conv2d(self.out, 512, 1, variables_collections=self.var_coll)
        self + tf.reshape(self.out, (-1, 512))

    def for_input(self, new_input):
        return TCML_ResNet_Omniglot_v2(new_input, self.name, self.deterministic_initialization, True)


def hr_res_net_tcml_Omniglot_builder_v2():
    return lambda x, name: TCML_ResNet_Omniglot_v2(x, name=name)




def hr_res_net_tcml_v1_builder():
    return lambda x, name: TCML_ResNet(x, name=name)


def hr_res_net_tcml_Omniglot_builder():
    return lambda x, name: TCML_ResNet_Omniglot(x, name=name)


if __name__ == '__main__':
    inp = tf.placeholder(tf.float32, (None, 84, 84, 3))
    net = TCML_ResNet(inp)
    print(net.out)
    print(far.hyperparameters())


