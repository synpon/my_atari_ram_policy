import tensorflow as tf
import numpy as np
from my_param_collection import ParamCollection
# from rl import Serializable
# from categorical import cat_sample, cat_entropy
# from ppo import PPOPolicy

# Helper functions.
def weight_variable(shape, stddev=0.1, initial=None):
    if initial is None:
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape, init_bias=0.1, initial=None):
    if initial is None:
        initial = tf.constant(init_bias, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

class AtariRAMPolicy(object): #PPOPolicy, Serializable):
    """
    TensorFlow policy to play Atari.
    adapted from cgt version in cs294 @ http://rll.berkeley.edu/deeprlcourse/
    """
    def __init__(self, n_actions):
        # Serializable.__init__(self, n_actions)
        n_in = 128
        n_hid = 64


        # Observations placeholder. batch_size samples with 128 features.
        self.o_no = tf.placeholder(tf.float64, shape=[None, n_in])
        # Actions
        self.a_n = tf.placeholder(tf.int8, shape=[None])
        # Rewards
        self.q_n = tf.placeholder(tf.float64, shape=[None])
        # Previous transition probability distribution.
        self.oldpdist_np = tf.placeholder(tf.float64, shape=[None, n_actions])
        # Relative importance of Kullback-Liebler Divergence to ultimate objective.
        self.lam = tf.placeholder(tf.float64)


        # Tack network dimensions to self so we can talk about them when
        # setting parameters.
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_actions = n_actions

        # Normalize observations.
        h0 = tf.div(tf.sub(self.o_no, 128.0), 128.0)

        # Initialize weights and bias from input to hidden layer.
        self.W_01 = weight_variable([n_in, n_hid])
        self.W_12 = weight_variable([n_hid, n_actions], stddev=0.01)
        self.b_01 = bias_variable([n_hid])
        self.b_12 = bias_variable([n_actions], init_bias=0.01)

        # Map input to hidden layer.
        h1 = tf.nn.tanh(tf.matmul(h0, self.W_01) + self.b_01)

        # Initialize weights and biases from hidden layer to action space.

        # Map hidden layer activations to probabilities of actions.
        self.probs_na = tf.nn.softmax(tf.matmul(h1, self.W_12) + self.b_12)

        logprobs_na = tf.log(self.probs_na)

        # This works.
        self.n_batch = tf.shape(self.a_n)[0]

        # Gather from a flattened version of the matrix since gather_nd does
        # not work on the gpu at this time.
        idx_flattened = tf.range(0, self.n_batch) * n_actions + tf.cast(self.a_n, tf.int32)

        # The modeled log probability of the choice taken for whole batch.
        logps_n = tf.gather(tf.reshape(logprobs_na, [-1]), idx_flattened)

        # Product of modeled log probability for chosen action and return.
        self.surr = tf.reduce_mean(tf.mul(logps_n , self.q_n))

        params = tf.trainable_variables()

        # Compute gradients of surrogate objective function.
        # self.surr_grads = tf.gradients(self.surr, [self.W_01, self.W_12, self.b_01, self.b_12])
        self.surr_grads = tf.gradients(self.surr, params)

        # Kullback-Liebler Divergence of new vs old transition probabilities.
        self.kl = tf.reduce_mean(
            tf.reduce_sum(
                tf.mul(self.oldpdist_np, tf.log(tf.div(self.oldpdist_np, self.probs_na))), 1))

        # Ultimate objective function for constrained optimization.
        penobj = tf.sub(self.surr, tf.mul(self.lam, self.kl))

        # Compute gradients of KLD-constrained objective function.
        self.penobj_grads = tf.gradients(penobj, params)

        # Attach a session with initialized variables to the class.
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

        self.pc = ParamCollection(self.sess, params)

    def step(self, X):
        feed_dict={
            self.o_no : X,
        }
        pdist_na = self.sess.run(self.probs_na, feed_dict=feed_dict)
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
        [surr_grads] = self.sess.run([self.surr_grads],feed_dict=feed_dict)
        return np.concatenate([p.flatten() for p in surr_grads],0)
        # return surr_grads

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
        [penobj_grads] = self.sess.run([self.penobj_grads], feed_dict=feed_dict)
        return np.concatenate([p.flatten() for p in penobj_grads],0)


    def compute_entropy(self, pdist_np):
        raise NotImplementedError
        # return cat_entropy(pdist_np)

    def get_parameters_flat(self):
        return self.pc.get_values_flat()
        # W_01 = self.sess.run(self.W_01)
        # W_12 = self.sess.run(self.W_12)
        # b_01 = self.sess.run(self.b_01)
        # b_12 = self.sess.run(self.b_12)
        # return np.concatenate([p.flatten() for p in [W_01, W_12, b_01, b_12]],0)

    def set_parameters_flat(self, th):
        return self.pc.set_values_flat(th)
        # self.sess.run(tf.initialize_all_variables())
        # n_in = self.n_in
        # n_hid = self.n_hid
        # n_actions = self.n_actions
        # W_01 = th[:n_hid*n_in].reshape(n_in,n_hid)
        # W_12 = th[n_hid*n_in:n_hid*n_in+n_hid*n_actions].reshape(n_hid,n_actions)
        # b_01 = th[-n_hid-n_actions:-n_actions]
        # b_12 = th[-n_actions:]
        # self.sess.run(tf.assign(self.W_01, W_01))
        # self.sess.run(tf.assign(self.W_12, W_12))
        # self.sess.run(tf.assign(self.b_01, b_01))
        # self.sess.run(tf.assign(self.b_12, b_12))
        # self.pc = ParamCollection(th)


def test_AtariRAMPolicy():
    """
    Test the model using some fake data.
    """
    # Make some dimensions for our fake observations, actions, and rewards.
    n_batch = 30000
    n_features = 128
    n_actions = 9
    lam = 1.0

    # Go ahead and initialize the policy.
    policy = AtariRAMPolicy(n_actions=n_actions)

    # Fake observations.
    obs = np.random.rand(n_batch, n_features)
    # Fake transition probabilities.
    probs_na = np.random.rand(n_batch, n_actions)
    # Fake actions.
    a_n = np.random.randint(0, n_actions, size=(n_batch,))
    # Fake rewards.
    q_n = np.random.rand(n_batch,)

    # Now for some tests.

    # # Test set_n_batch()
    # policy.set_n_batch(n_batch)
    # print(policy.n_batch)

    # testing get_parameters_flat()
    th = policy.get_parameters_flat() + 1.
    #
    # testing set_parameters_flat()
    policy.set_parameters_flat(th)

    # testing compute_gradient()
    policy.compute_gradient(probs_na, obs, a_n, q_n)

    # testing compute_surr_kl()
    policy.compute_surr_kl(probs_na, obs, a_n, q_n)

    # testing compute_grad_lagrangian()
    policy.compute_grad_lagrangian(lam, probs_na, obs, a_n, q_n)

    # Make sure we still have the same parameters
    th_new = policy.get_parameters_flat()

    # assert not np.any(th_new - th)
    print(th_new - th)


if __name__ == "__main__":
    test_AtariRAMPolicy()
