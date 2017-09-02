import numpy as np
import tensorflow as tf
import sys
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


def make_mnist(noise_type):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        y_train = mnist.train.images
        x_train = []
        # Apply Noise to each Image
        for img in y_train:
                x_train.append(apply_noise(noise_type, img))
        y_train = np.array(y_train)

        y_test = mnist.test.images
        x_test = []
        # Apply Noise to each Image
        for img in y_test:
                x_test.append(apply_noise(noise_type, img))
        y_test = np.array(y_test)

        return (x_train, y_train, x_test, y_test)

def make_sine(noise_type, num, window):

        train_size = int((num-window)*0.8)
        n = np.linspace(0, 20.0, num)

        # Make the sine wave...
        s = np.sin(2*np.pi*n)
        s_n = apply_noise(noise_type, s)

        n_data = make_data(s_n, window)
        data = make_data(s, window)

        x_train = n_data[0:train_size,:]
        y_train = data[0:train_size,:]
        x_test = n_data[train_size:,:]
        y_test = data[train_size:,:]

        return (x_train, y_train, x_test, y_test)

def apply_noise(noise_type, s):
        num = len(s)
        # Create the Noise
        # None
        if noise_type == 0:
                s_n = s

        # Additive Isotropic Gaussian Noise
        if noise_type == 1:
                n_level = 0.2
                noise = np.random.normal(0, n_level, num)
                s_n = s + noise

                # Calculate SNR
                p_noise = 1/len(noise)*np.sum(np.square(np.abs(noise)))
                p_sn = 1/len(s_n)*np.sum(np.square(np.abs(s_n)))
                snr = 10*np.log10((p_sn - p_noise)/p_noise)

                print("SNR(dB): %.2f" % (snr))

        # Masking Noise : Some fraction of s are set to zero
        if noise_type == 2:
                mask_perc = 0.2
                idx = np.arange(num)
                s_idx = np.random.choice(idx, int(num*mask_perc), replace=False)
                s_n = np.zeros(len(s))
                np.copyto(s_n, s)
                s_n[s_idx] = 0

        # Salt and Pepper Nosie : Some fraction of s are set to either min or max based on a coin flip
        if noise_type == 3:
                s_max = np.max(s)
                s_min = np.min(s)
                mask_perc = 0.2
                idx = np.arange(num)
                s_idx = np.random.choice(idx, int(num*mask_perc), replace=False)
                s_n = np.zeros(len(s))
                np.copyto(s_n, s)
                flip = np.random.random_sample(len(s_idx))
                s_n[s_idx[flip > 0.5]] = s_max
                s_n[s_idx[flip < 0.5]] = s_min

        return s_n

def display_digit(img, lbl):
    label = lbl.argmax(axis=0)
    image = img.reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def print_weights(data_path, n_weights):
        for w in n_weights:
                weights = n_weights[w].eval()
                shp = weights.shape
                f = open(data_path + str(w) + ".h", "w")
                f.write("weight_t " + str(w) + "[" + str(shp[0]) + "][" + str(shp[1]) + "] = {{")
                for i in range(shp[0]-1):
                        for j in range(shp[1]-1):
                                f.write(str(weights[i][j]) + ",")
                        f.write(str(weights[i][shp[1]-1]) + "},\n{")
                for j in range(shp[1]-1):
                        f.write(str(weights[shp[0]-1][j]) + ",")
                f.write(str(weights[shp[0]-1][shp[1]-1]) + "}};\n")
                f.flush()
                f.close()

def print_biases(data_path, n_biases):
        for b in n_biases:
                biases = n_biases[b].eval()
                shp = biases.shape
                f = open(data_path + str(b) + ".h", "w")
                f.write("bias_t " + str(b) + "[" + str(shp[0]) + "] = {")
                for j in range(shp[0]-1):
                        f.write(str(biases[j]) + ",")
                f.write(str(biases[shp[0]-1]) + "};\n")
                f.flush()
                f.close()

def  make_data(data, window_size):
        X = []

        win_size = window_size
        for i in range(len(data)-win_size):
                X.append(data[i:i+win_size])

        return np.array(X)

# Dataset Type
#  0 : Sine
#  1 : MNIST
dataset_type = 0

# Noise Types:
#  0 : None
#  1 : Additive Isotripic Gaussian
#  2 : Maksing
#  3 : Salt and Pepper
noise_type = 0

if dataset_type == 0:
        # AutoEncoder will look like this:
        # * Layer1: 64
        # * Layer2: 32
        # * Layer3: 16
        # * Layer4: 16
        # * Layer5: 32
        # * Layer6: 64
        Layer_1 = 64
        Layer_2 = 32
        Layer_3 = 16
        window = Layer_1
        num = 2000
        (x_train, y_train, x_test, y_test) = make_sine(noise_type, num, window)
elif dataset_type == 1:
        # AutoEncoder will look like this:
        # * Layer1: 784
        # * Layer2: 400
        # * Layer3: 200
        # * Layer4: 200
        # * Layer5: 400
        # * Layer6: 784
        Layer_1 = 784
        Layer_2 = 600
        Layer_3 = 300
        (x_train, y_train, x_test, y_test) = make_mnist(noise_type)


n_weights = {
        'w_l1' : tf.Variable(tf.truncated_normal([Layer_1,Layer_1], stddev=0.1), name="w_l1"),
        'w_l2' : tf.Variable(tf.truncated_normal([Layer_1,Layer_2], stddev=0.1), name="w_l2"),
        'w_l3' : tf.Variable(tf.truncated_normal([Layer_2,Layer_3], stddev=0.1), name="w_l3"),
        'w_l4' : tf.Variable(tf.truncated_normal([Layer_3,Layer_3], stddev=0.1), name="w_l4"),
        'w_l5' : tf.Variable(tf.truncated_normal([Layer_3,Layer_2], stddev=0.1), name="w_l5"),
        'w_l6' : tf.Variable(tf.truncated_normal([Layer_2,Layer_1], stddev=0.1), name="w_l6"),
}

n_biases = {
        'b_l1' : tf.Variable(tf.constant(0.1, shape=[Layer_1]), name="b_l1"),
        'b_l2' : tf.Variable(tf.constant(0.1, shape=[Layer_2]), name="b_l2"),
        'b_l3' : tf.Variable(tf.constant(0.1, shape=[Layer_3]), name="b_l3"),
        'b_l4' : tf.Variable(tf.constant(0.1, shape=[Layer_3]), name="b_l4"),
        'b_l5' : tf.Variable(tf.constant(0.1, shape=[Layer_2]), name="b_l5"),
        'b_l6' : tf.Variable(tf.constant(0.1, shape=[Layer_1]), name="b_l6"),
}

x = tf.placeholder(tf.float32, [None, Layer_1])

y = tf.placeholder(tf.float32, [None, Layer_1])

# Layer 1
a_l1 = tf.matmul(x, n_weights["w_l1"]) + n_biases["b_l1"]
r_l1 = tf.nn.relu(a_l1)

# Layer 2
a_l2 = tf.matmul(r_l1, n_weights["w_l2"]) + n_biases["b_l2"]
r_l2 = tf.nn.relu(a_l2)

# Layer 3
a_l3 = tf.matmul(r_l2, n_weights["w_l3"]) + n_biases["b_l3"]
r_l3 = tf.nn.relu(a_l3)

# Layer 4
a_l4 = tf.matmul(r_l3, n_weights["w_l4"]) + n_biases["b_l4"]
r_l4 = tf.nn.relu(a_l4)

# Layer 5
a_l5 = tf.matmul(r_l4, n_weights["w_l5"]) + n_biases["b_l5"]
r_l5 = tf.nn.relu(a_l5)

# Layer 6
pred = tf.matmul(r_l5, n_weights["w_l6"]) + n_biases["b_l6"]

# Cost Function
loss = tf.reduce_mean(tf.squared_difference(y, pred))


with tf.Session() as sess:
        trainer = tf.train.AdamOptimizer(1e-4)
        train_step = trainer.minimize(loss)

        sess.run(tf.global_variables_initializer())


        # Use the entire data set as the batch
        # x_train = data[0:train_size,:]
        # y_train = x_train

        # Standard Denoising using the uncorrupted signals in the loss functions
        for i in range(10000):
                if i % 200 == 0:
                        p_train_val = sess.run([loss], feed_dict={x: x_train, y: y_train})
                        print('step: %d, loss: %.8f' % (i, p_train_val[0]))
                train_step.run(feed_dict={x: x_train, y: y_train})


        p_pred_n = sess.run([pred], feed_dict={x: x_test, y: y_test})[0]
        print("MSE(Denoise): ", np.mean(np.square(p_pred_n - y_test)))
        print("MSE(Vs Corrupted): ", np.mean(np.square(p_pred_n - x_test)))

        data_path = "data/"

        # Print the Dataset to a file
        np.savetxt(data_path + 'data.out', x_test, delimiter=',')

        # Print out the expected predictions
        np.savetxt(data_path + 'expected.out', p_pred_n, delimiter=',')

        # Save the Weight to a file
        print_weights(data_path, n_weights)

        # Save the Biases to a file
        print_biases(data_path, n_biases)

        # Display some Results
        out_n = p_pred_n[0].reshape([28,28])
        exp = y_test[0].reshape([28,28])
        exp_n = x_test[0].reshape([28,28])

        fig = figure(1)

        ax1 = fig.add_subplot(311)
        ax1.imshow(exp, cmap=plt.get_cmap('gray_r'))
        ax1.grid(True)

        ax2 = fig.add_subplot(312)
        ax2.imshow(out_n, cmap=plt.get_cmap('gray_r'))
        ax2.grid(True)

        ax3 = fig.add_subplot(313)
        ax3.imshow(exp_n, cmap=plt.get_cmap('gray_r'))
        ax3.grid(True)

        show()
