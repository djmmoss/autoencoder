import numpy as np
import tensorflow as tf
import sys
from scipy import signal
import scipy
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
        a = 1
        w = 2*np.pi
        wt = w*n
        p_phase = w*(n -  np.floor(n))

        i = a*np.cos(0)
        q = a*np.sin(0)
        s = i*np.cos(wt) - q*np.sin(wt)

        #s = a*np.cos(2*np.pi*n)

        s_n = apply_noise(noise_type, s)

        n_data = make_data(s_n, window)
        data = make_data(s, window)

        x_train = n_data[0:train_size,:]
        y_train = data[0:train_size,:]
        x_test = n_data[train_size:,:]
        y_test = data[train_size:,:]

        return (x_train, y_train, x_test, y_test)

def make_carrier(noise_type):

        fs = 5e3
        N = 1e6
        amp = 2 * np.sqrt(2)
        time = np.arange(N) / float(fs)
        mod = 500*np.cos(2*np.pi*0.25*time)
        #carrier = amp * np.sin(2*np.pi*3e3*time + mod)
        carrier = amp * np.sin(2*np.pi*2e3*time + mod) + amp*np.sin(2*np.pi*1e3*time)
        x = carrier

        # Blanket the Entire Signal With Noise
        #noise_power = 0.001 * fs / 2
        #noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        #x_n += noise

        # High Period of Noise for some time
        noise_power = 0.01 * fs / 2
        noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        n_start = int(len(time)*0.2)
        n_end = int(len(time)*0.4)
        noise[:n_start] *= 0
        noise[n_end:] *= 0
        x_n = carrier + noise

        # Rogue Signal for sometime
        #noise = amp * np.sin(2*np.pi*5e2*time)
        #n_start = int(len(time)*0.6)
        #n_end = int(len(time)*0.7)
        #noise[:n_start] *= 0
        #noise[n_end:] *= 0
        #x_n = carrier + noise


        f, t, Sxx = signal.spectrogram(x, fs, nperseg=63, nfft=63)
        f_n, t_n, Sxx_n = signal.spectrogram(x_n, fs, nperseg=63, nfft=63)

        #s_n = apply_noise(noise_type, s)

        #n_data = make_data(s_n, window)
        #data = make_data(s, window)

        #fig = figure(1)

        #ax1 = fig.add_subplot(211)
        #ax1.pcolormesh(t, f, Sxx)

        #ax2 = fig.add_subplot(212)
        #ax2.pcolormesh(t_n, f_n, Sxx_n)

        #show()

        data = Sxx.T
        n_data = data
        train_size = int(len(data)*0.8)

        x_train = n_data[0:train_size,:]
        y_train = data[0:train_size,:]
        x_test = n_data[train_size:,:]
        y_test = data[train_size:,:]

        return (x_train, y_train, x_test, y_test, f, t, data, Sxx_n.T)

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
                f.write("static weight_t " + str(w) + "[" + str(shp[0]) + "][" + str(shp[1]) + "] = {{")
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
                f.write("static bias_t " + str(b) + "[" + str(shp[0]) + "] = {")
                for j in range(shp[0]-1):
                        f.write(str(biases[j]) + ",")
                f.write(str(biases[shp[0]-1]) + "};\n")
                f.flush()
                f.close()

def print_network(L1, L2, L3):
    f = open(data_path + "net.h", "w")
    f.write("#define LAYER_1 " + str(L1) + "\n")
    f.write("#define LAYER_2 " + str(L2) + "\n")
    f.write("#define LAYER_3 " + str(L3) + "\n")
    f.flush()
    f.close()

    f = open(data_path + "noc_block_autoenc_tb.vh", "w")
    f.write("`define TEST_SPP " + str(L1) + "\n")
    f.write("`define TEST_TRL 2\n")
    f.write("`define TEST_ERR 32\n")
    f.flush()
    f.close()

def  make_data(data, window_size):
        X = []

        win_size = int(window_size)
        for i in range(len(data)-win_size):
                win = data[i:i+win_size]
                fft_win = np.fft.fft(win)
                r_fft_win = np.real(fft_win)
                i_fft_win = np.imag(fft_win)
                c_fft_win = np.concatenate((r_fft_win, i_fft_win))
                X.append(win)
                #X.append(c_fft_win)

        return np.array(X)

# Dataset Type
#  0 : Sine
#  1 : MNIST
#  2 : Carrier
dataset_type = 2

# Noise Types:
#  0 : None
#  1 : Additive Isotripic Gaussian
#  2 : Masking
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
        Layer_1 = 32
        Layer_2 = 16
        Layer_3 = 8
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
elif dataset_type == 2:
        # * Layer1: 64
        # * Layer2: 32
        # * Layer3: 16
        # * Layer4: 16
        # * Layer5: 32
        # * Layer6: 64
        Layer_1 = 32
        Layer_2 = 16
        Layer_3 = 8
        (x_train, y_train, x_test, y_test, f, t, dat, n_dat) = make_carrier(noise_type)


n_weights = {
        #'w_l1' : tf.Variable(tf.truncated_normal([Layer_1,Layer_1], stddev=0.1), name="w_l1"),
        'w_l2' : tf.Variable(tf.truncated_normal([Layer_1,Layer_2], stddev=0.1), name="w_l2"),
        'w_l3' : tf.Variable(tf.truncated_normal([Layer_2,Layer_3], stddev=0.1), name="w_l3"),
        #'w_l4' : tf.Variable(tf.truncated_normal([Layer_3,Layer_3], stddev=0.1), name="w_l4"),
        'w_l5' : tf.Variable(tf.truncated_normal([Layer_3,Layer_2], stddev=0.1), name="w_l5"),
        'w_l6' : tf.Variable(tf.truncated_normal([Layer_2,Layer_1], stddev=0.1), name="w_l6"),
}

n_biases = {
        #'b_l1' : tf.Variable(tf.constant(0.1, shape=[Layer_1]), name="b_l1"),
        'b_l2' : tf.Variable(tf.constant(0.1, shape=[Layer_2]), name="b_l2"),
        'b_l3' : tf.Variable(tf.constant(0.1, shape=[Layer_3]), name="b_l3"),
        #'b_l4' : tf.Variable(tf.constant(0.1, shape=[Layer_3]), name="b_l4"),
        'b_l5' : tf.Variable(tf.constant(0.1, shape=[Layer_2]), name="b_l5"),
        'b_l6' : tf.Variable(tf.constant(0.1, shape=[Layer_1]), name="b_l6"),
}

x = tf.placeholder(tf.float32, [None, Layer_1])

y = tf.placeholder(tf.float32, [None, Layer_1])

# Layer 1
#a_l1 = tf.matmul(x, n_weights["w_l1"]) + n_biases["b_l1"]
#r_l1 = tf.nn.relu(a_l1)

# Layer 2
a_l2 = tf.matmul(x, n_weights["w_l2"]) + n_biases["b_l2"]
r_l2 = tf.nn.relu(a_l2)

# Layer 3
a_l3 = tf.matmul(r_l2, n_weights["w_l3"]) + n_biases["b_l3"]
r_l3 = tf.nn.relu(a_l3)

# Layer 4
#a_l4 = tf.matmul(r_l3, n_weights["w_l4"]) + n_biases["b_l4"]
#r_l4 = tf.nn.relu(a_l4)

# Layer 5
a_l5 = tf.matmul(r_l3, n_weights["w_l5"]) + n_biases["b_l5"]
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
        for i in range(20000):
                if i % 200 == 0:
                        p_train_val = sess.run([loss], feed_dict={x: x_train, y: y_train})
                        print('step: %d, loss: %.8f' % (i, p_train_val[0]))
                train_step.run(feed_dict={x: x_train, y: y_train})


        p_pred_n = sess.run([pred], feed_dict={x: x_test})[0]
        print("MSE(Denoise): ", np.mean(np.square(p_pred_n - y_test)))
        print("MSE(Vs Corrupted): ", np.mean(np.square(p_pred_n - x_test)))

        #data_path = "data/"

        l2norm = np.sum(np.square(p_pred_n - x_test), 1)

        # Print the Dataset to a file
        #np.savetxt(data_path + 'data.out', x_test[:,0], delimiter=',')

        # Print out the expected predictions
        #np.savetxt(data_path + 'expected.out', l2norm, delimiter=',')

        # Save the Weight to a file
        #print_weights(data_path, n_weights)

        # Save the Biases to a file
        #print_biases(data_path, n_biases)

        #print_network(Layer_1, Layer_2, Layer_3)

        # Display some Results
        p_pred = sess.run([pred], feed_dict={x: n_dat})[0]
        l2norm = np.sum(np.square(p_pred - n_dat), 1)

        fig = figure(1)

        ax1 = fig.add_subplot(311)
        ax1.pcolormesh(t, f, p_pred.T)

        ax2 = fig.add_subplot(312)
        ax2.pcolormesh(t, f, n_dat.T)

        ax3 = fig.add_subplot(313)
        ax3.plot(t, l2norm)
        ax3.margins(x=0,y=0)

        fig.savefig("noise_1.png")
        show()
