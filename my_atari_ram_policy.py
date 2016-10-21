import tensorflow as tf
import numpy as np
from param_collection import ParamCollection
from rl import Serializable
from categorical import cat_sample, cat_entropy
from ppo import PPOPolicy


def weight_variable(shape, stddev=0.1, initial=None):
    if initial is None:
        initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, init_bias=0.1,initial=None):
    if initial is None:
        initial = tf.constant(init_bias, shape=shape)
    return tf.Variable(initial)

class AtariRAMPolicy(PPOPolicy, Serializable):
    def __init__(self, n_actions):
        Serializable.__init__(self, n_actions)
        n_in = 128
        n_hid = 64

        # Observations placeholder. batch_size samples with 128 features.
        self.o_no = tf.placeholder(tf.float32, shape=[None, n_in])
        self.a_n = tf.placeholder(tf.int8, shape=[None])
        self.q_n = tf.placeholder(tf.float32, shape=[None])
        self.oldpdist_np = tf.placeholder(tf.float32, shape=[None, n_actions])
        self.lam = tf.placeholder(tf.float32, shape=[1])

        # Now tack them to self so we can talk about them to feed_dict.
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_actions = n_actions

        # Normalize observations.
        h0 = tf.div(tf.sub(self.o_no, 128.0), 128.0)

        # Initialize weights and bias from input to hidden layer.
        self.W_01 = weight_variable([n_in, n_hid])
        self.b_01 = bias_variable([n_hid])

        # Map input to hidden layer.
        h1 = tf.nn.tanh(tf.matmul(h0, self.W_01) + self.b_01)

        # Initialize weights and biases from hidden layer to action space.
        self.W_12 = weight_variable([n_hid, n_actions], stddev=0.01)
        self.b_12 = bias_variable([n_actions], init_bias=0.01)

        # Map hidden layer activations to probabilities of actions.
        probs_na = tf.nn.softmax(tf.matmul(h1, self.W_12) + self.b_12)

        logprobs_na = tf.log(probs_na)

        # Gather from a flattened version of the matrix since gather_nd does
        # not work on the gpu at this time.
        self.batch_size = logprobs_na.get_shape().as_list()[0]
        idx_flattened = tf.range(0, self.batch_size) * n_actions + tf.cast(self.a_n, tf.int32)

        # The modeled log probability of the choice taken for whole batch.
        logps_n = tf.gather(tf.reshape(logprobs_na, [-1]), idx_flattened)

        # Product of modeled log probability for chosen action and return.
        surr = tf.reduce_mean(tf.mul(logps_n , self.q_n))

        # Compute gradients of surrogate objective function.
        surr_grads = tf.gradients(surr, [self.W_01, self.W_12, self.b_01, self.b_12])

        # Kullback-Liebler Divergence of new vs old transition probabilities.
        kl = tf.reduce_mean(
            tf.reduce_sum(
                tf.mul(self.oldpdist_np, tf.log(tf.div(self.oldpdist_np, probs_na))), 1))

        # Ultimate objective function for constrained optimization.
        penobj = tf.sub(surr, tf.mul(self.lam, kl))

        # Compute gradients of KLD-constrained objective function.
        penobj_grads = tf.gradients(penobj, [self.W_01, self.W_12, self.b_01, self.b_12])

        self.f_probs = probs_na
        self.surr = surr
        self.surr_grads = surr_grads
        self.kl = kl
        self.penobj_grads = penobj_grads
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def step(self, X):
        feed_dict={
            self.o_no : X,
        }
        pdist_na = self.sess.run(self.f_probs,feed_dict=feed_dict)
        # pdist_na = self.f_probs(X)
        acts_n = cat_sample(pdist_na)
        return {
            "action" : acts_n,
            "pdist" : pdist_na
        }

    def compute_gradient(self, pdist_np, o_no, a_n, q_n):
        feed_dict={
            self.oldpdist_np : pdist_np,
            self.o_no : o_no,
            self.a_n : a_n,
            self.q_n : q_n
        }
        surr_grads = self.sess.run([self.surr_grads],feed_dict=feed_dict)
        return np.concatenate([p.flatten() for p in surr_grads],0)

    def compute_surr_kl(self, pdist_np, o_no, a_n, q_n):
        feed_dict={
            self.oldpdist_np : pdist_np,
            self.o_no : o_no,
            self.a_n : a_n,
            self.q_n : q_n
        }
        surr, kl = self.sess.run([self.surr, self.kl],feed_dict=feed_dict)
        return surr, kl

    def compute_grad_lagrangian(self, lam, pdist_np, o_no, a_n, q_n):
        feed_dict={
            self.lam : lam,
            self.oldpdist_np : pdist_np,
            self.o_no : o_no,
            self.a_n : a_n,
            self.q_n : q_n
        }
        penobj_grads = self.sess.run([self.penobj_grads], feed_dict=feed_dict)
        return np.concatenate([p.flatten() for p in penobj_grads])

    def compute_entropy(self, pdist_np):
        return cat_entropy(pdist_np)

    def get_parameters_flat(self):
        W_01 = self.sess.run(self.W_01)
        W_12 = self.sess.run(self.W_12)
        b_01 = self.sess.run(self.b_01)
        b_12 = self.sess.run(self.b_12)
        return np.concatenate([p.flatten() for p in [W_01, W_12, b_01, b_12]],0)

    def set_paramters_flat(self, th):
        n_in = self.n_in
        n_hid = self.n_hid
        n_actions = self.n_actions
        W_01 = tf.Variable(th[:n_hid*n_in].reshape(n_in,n_hid))
        W_12 = tf.Variable(th[n_hid*n_in:n_hid*n_in+n_hid*n_actions].reshape(n_hid,n_in))
        b_01 = tf.Variable(th[-n_hid-n_actions:])
        b_12 = tf.Variable(th[-n_actions:])
        self.sess.run(tf.assign(self.W_01, W_01))
        self.sess.run(tf.assign(self.W_12, W_12))
        self.sess.run(tf.assign(self.b_01, b_01))
        self.sess.run(tf.assign(self.b_12, b_12))
