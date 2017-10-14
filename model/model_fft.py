import numpy as np
import tensorflow as tf
import sys
from scipy import signal
import scipy
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

def noise_band(n_l, amp, fs=5e3, N=1e4):
        time = np.arange(N) / float(fs)
        noise_power = n_l * amp * fs / 2
        noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        n_start = int(len(time)*0.2)
        n_end = int(len(time)*0.4)
        noise[:n_start] *= 0
        noise[n_end:] *= 0
        return (noise, n_start, n_end)

def noise_rogue(n_l, amp, fs=5e3, N=1e4):
        # Rogue Signal for sometime
        time = np.arange(N) / float(fs)
        n_mod = 25*np.cos(2*np.pi*5*time)
        noise = n_l * amp * np.sin(2*np.pi*5e2*time + n_mod)
        n_start = int(len(time)*0.6)
        n_end = int(len(time)*0.7)
        noise[:n_start] *= 0
        noise[n_end:] *= 0
        return (noise, n_start, n_end)

def noise_tamper(n_l, amp, fs=5e3, N=1e4):
        # Tampering with the main carrier
        time = np.arange(N) / float(fs)
        n_mod = 25*np.cos(2*np.pi*5*time)
        noise = -n_l * amp * np.sin(2*np.pi*2e3*time + n_mod) # Cancel
        n_start = int(len(time)*0.4)
        n_end = int(len(time)*0.55)
        noise[:n_start] *= 0
        noise[n_end:] *= 0
        return (noise, n_start, n_end)

def noise_repeat(n_l, amp, fs=5e3, N=1e4):
        # Repeat Carrier at a very close frequency
        time = np.arange(N) / float(fs)
        noise = n_l*amp * np.sin(2*np.pi*1.9e3*time)
        n_start = int(len(time)*0.3)
        n_end = int(len(time)*0.4)
        noise[:n_start] *= 0
        noise[n_end:] *= 0
        return (noise, n_start, n_end)

def plot_spectrum(t, f, p_pred, n_dat, l2norm, filename):
        fig = figure(1)

        ax1 = fig.add_subplot(311)
        ax1.pcolormesh(t, f, p_pred.T)

        ax2 = fig.add_subplot(312)
        ax2.pcolormesh(t, f, n_dat.T)

        ax3 = fig.add_subplot(313)
        ax3.plot(t, l2norm)
        ax3.margins(x=0,y=0)

        fig.savefig(filename + ".png")
        show()

def calc_snr(noise, signal):
            # Calculate SNR
            p_noise = 1/len(noise)*np.sum(np.square(np.abs(noise)))
            p_sn = 1/len(signal)*np.sum(np.square(np.abs(signal)))
            return 10*np.log10((p_sn - p_noise)/p_noise)

def carrier(amp,fs=5e3, N=1e4):
        time = np.arange(N) / float(fs)
        noise = np.random.normal(scale=0.001, size=time.shape)
        mod = 25*np.cos(2*np.pi*5*time) + noise
        i = amp*np.cos(mod)
        q = amp*np.sin(mod)
        #c_test = amp * np.cos(2*np.pi*2e3*time + mod)
        # c = i*np.cos(2*np.pi*2e3*time) - q*np.sin(2*np.pi*2e3*time)
        return (i, q)

def make_carrier(amp, fs, N, window):
        t, s = carrier(amp, fs, N)


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

def print_fft_w(L1):
    f = open(data_path + "fft_w.h", "w")
    f.write("static cplx fft_w [" + str(L1) + "] = {")
    for i in range(1, L1):
        val = np.exp(2j*np.pi*i*(i/L1))
        f.write("{ " + str(np.real(val)) + ", " + str(np.imag(val)) + "},\n")

    val = np.exp(2j*np.pi*L1*(L1/L1))
    f.write("{ " + str(np.real(val)) + ", " + str(np.imag(val)) + "}};")
    f.flush()
    f.close()


def  make_data(data, window_size):
        X = []

        win_size = int(window_size)
        for i in range(len(data)-win_size):
                win = data[i:i+win_size]
                X.append(win)

        return np.array(X)

# * Layer1: 32
# * Layer2: 16
# * Layer3: 8
# * Layer4: 16
# * Layer5: 32
Layer_1 = 32
Layer_2 = 16
Layer_3 = 8

fs = 5#e3
N = 1e2
amp = 2 * np.sqrt(2)

i_s, q_s = carrier(amp, fs, N)

sig = i_s + q_s*1j

sig_window = make_data(sig, Layer_1/2)

data = []
for sig_w in sig_window:
    fft_res = np.fft.fft(sig_w)
    fft_r = np.real(fft_res)
    fft_i = np.imag(fft_res)
    data.append(np.concatenate((fft_r, fft_i)))

data = np.array(data)
n_data = data
train_size = int(len(data)*0.8)

x_train = n_data[0:train_size,:]
y_train = data[0:train_size,:]
x_test = n_data[train_size:,:]
y_test = data[train_size:,:]


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

        loss_rate = []
        step_rate = []

        # Standard Denoising using the uncorrupted signals in the loss functions
        for i in range(10000):
                if i % 200 == 0:
                        p_train_val = sess.run([loss], feed_dict={x: x_train, y: y_train})
                        print('step: %d, loss: %.8f' % (i, p_train_val[0]))
                        step_rate.append(i)
                        loss_rate.append(p_train_val[0])
                train_step.run(feed_dict={x: x_train, y: y_train})

        p_pred_n = sess.run([pred], feed_dict={x: x_test})[0]
        print("MSE(Denoise): ", np.mean(np.square(p_pred_n - y_test)))
        print("MSE(Vs Corrupted): ", np.mean(np.square(p_pred_n - x_test)))
        l2norm = np.sum(np.square(p_pred_n - x_test), 1)

        data_path = "data/"
        np.savetxt(data_path + 'data_i.out', i_s, delimiter=',')
        np.savetxt(data_path + 'data_q.out', q_s, delimiter=',')
        np.savetxt(data_path + 'expected.out', l2norm, delimiter=',')
        print_weights(data_path, n_weights)
        print_biases(data_path, n_biases)
        print_network(Layer_1, Layer_2, Layer_3)
        print_fft_w(int(Layer_1/2))

        p_pred = sess.run([pred], feed_dict={x: data})[0]

        diff = int(Layer_1/2)
        i_s = i_s[:-diff]
        q_s = q_s[:-diff]
        t = np.linspace(0, 1, len(i_s))
        c = i_s*np.cos(2*np.pi*2e3*t) - q_s*np.sin(2*np.pi*2e3*t)
        f_sp, t_sp, Sxx = signal.spectrogram(c, fs, window='blackmanharris', nperseg=63, noverlap=62)

        i_p = []
        q_p = []
        for win in p_pred:
            ifft_w = win[:diff] + win[diff:]*1j
            ifft_res = np.fft.ifft(ifft_w)
            i_p.append(np.real(ifft_res[0]))
            q_p.append(np.imag(ifft_res[0]))

        i_p = np.array(i_p)
        q_p = np.array(q_p)
        p = i_p*np.cos(2*np.pi*2e3*t) - q_p*np.sin(2*np.pi*2e3*t)

        fig = figure(1)

        ax1 = fig.add_subplot(311)
        ax1.plot(t, c)
        ax1.margins(x=0,y=0)

        ax2 = fig.add_subplot(312)
        ax2.pcolormesh(t_sp, f_sp, Sxx)

        ax2 = fig.add_subplot(313)
        ax2.plot(t, p)
        ax2.margins(x=0,y=0)

        show()



        """
        noise_level = np.linspace(0.00001, 0.002, 100)
        noise_level = np.concatenate((noise_level, np.linspace(0.002, 0.2, 100)))
        gb_f_snr = []
        gb_f_l2n = []

        for n_l in noise_level:
            noise, n_start, n_end = noise_band(n_l, 0.001, fs, N)
            x_n = c + noise

            snr = calc_snr(noise[n_start:n_end], x_n[n_start:n_end])

            #f_n, t_n, Sxx_n = signal.spectrogram(x_n, fs, window='blackmanharris', nperseg=63, noverlap=62)
            #n_dat = Sxx_n.T

            n_dat = make_data(x_n, window)

            # Display some Results
            p_pred = sess.run([pred], feed_dict={x: n_dat})[0]
            l2norm = np.sum(np.square(p_pred - n_dat), 1)
            gb_f_snr.append(snr)
            gb_f_l2n.append(np.mean(l2norm[n_start:n_end])/baseline)

            #print("%.3f - SNR(dB): %.2f, L2-Norm: %.4f, Baseline: %.4f" % (n_l, snr, np.mean(l2norm[n_start:n_end]), np.mean(l2norm[n_end:])))


        noise_level = np.linspace(0.1, 1, 100)
        noise_level = np.concatenate((noise_level, np.linspace(1, 21, 100)))
        rs_f_snr = []
        rs_f_l2n = []

        for n_l in noise_level:
            noise, n_start, n_end = noise_rogue(n_l, amp, fs, N)
            x_n = c + noise

            snr = calc_snr(noise[n_start:n_end], x_n[n_start:n_end])

            #f_n, t_n, Sxx_n = signal.spectrogram(x_n, fs, window='blackmanharris', nperseg=63, noverlap=62)
            #n_dat = Sxx_n.T

            n_dat = make_data(x_n, window)

            # Display some Results
            p_pred = sess.run([pred], feed_dict={x: n_dat})[0]
            l2norm = np.sum(np.square(p_pred - n_dat), 1)
            rs_f_snr.append(snr)
            rs_f_l2n.append(np.mean(l2norm[n_start:n_end])/baseline)

            #print("%.3f - SNR(dB): %.2f, L2-Norm: %.4f, Baseline: %.4f" % (n_l, snr, np.mean(l2norm[n_start:n_end]), np.mean(l2norm[n_end:])))

        noise_level = np.linspace(0.1, 0.999999, 200)

        tc_f_snr = []
        tc_f_l2n = []

        for n_l in noise_level:
            noise, n_start, n_end = noise_tamper(n_l, amp, fs, N)
            x_n = c + noise

            snr = calc_snr(noise[n_start:n_end], x_n[n_start:n_end])

            #f_n, t_n, Sxx_n = signal.spectrogram(x_n, fs, window='blackmanharris', nperseg=63, noverlap=62)
            #n_dat = Sxx_n.T

            n_dat = make_data(x_n, window)

            # Display some Results
            p_pred = sess.run([pred], feed_dict={x: n_dat})[0]
            l2norm = np.sum(np.square(p_pred - n_dat), 1)
            tc_f_snr.append(snr)
            tc_f_l2n.append(np.mean(l2norm[n_start:n_end])/baseline)

            #print("%.3f - SNR(dB): %.2f, L2-Norm: %.4f, Baseline: %.4f" % (n_l, snr, np.mean(l2norm[n_start:n_end]), np.mean(l2norm[n_end:])))

        noise_level = np.linspace(0.1, 1, 100)
        noise_level = np.concatenate((noise_level, np.linspace(1, 21, 100)))

        rc_f_snr = []
        rc_f_l2n = []

        for n_l in noise_level:
            noise, n_start, n_end = noise_repeat(n_l, amp, fs, N)
            x_n = c + noise

            snr = calc_snr(noise[n_start:n_end], x_n[n_start:n_end])

            #f_n, t_n, Sxx_n = signal.spectrogram(x_n, fs, window='blackmanharris', nperseg=63, noverlap=62)
            #n_dat = Sxx_n.T

            n_dat = make_data(x_n, window)

            # Display some Results
            p_pred = sess.run([pred], feed_dict={x: n_dat})[0]
            l2norm = np.sum(np.square(p_pred - n_dat), 1)
            rc_f_snr.append(snr)
            rc_f_l2n.append(np.mean(l2norm[n_start:n_end])/baseline)

            #print("%.3f - SNR(dB): %.2f, L2-Norm: %.4f, Baseline: %.4f" % (n_l, snr, np.mean(l2norm[n_start:n_end]), np.mean(l2norm[n_end:])))

        gb_f_snr = np.array(gb_f_snr)
        gb_f_l2n = np.array(gb_f_l2n)

        rs_f_snr = np.array(rs_f_snr)
        rs_f_l2n = np.array(rs_f_l2n)

        tc_f_snr = np.array(tc_f_snr)
        tc_f_l2n = np.array(tc_f_l2n)

        rc_f_snr = np.array(rc_f_snr)
        rc_f_l2n = np.array(rc_f_l2n)

        plt.semilogx(gb_f_l2n, gb_f_snr, label="Gaussian Band")
        plt.semilogx(rs_f_l2n, rs_f_snr, label="Rogue Signal")
        plt.semilogx(tc_f_l2n, tc_f_snr, label="Carrier Tampering")
        plt.semilogx(rc_f_l2n, rc_f_snr, label="Repeat Carrier f shifted")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)
        plt.ylim((-20, 20))
        plt.margins(x=0,y=0)
        plt.xlabel("Number of Times larger than the Basline")
        plt.ylabel("SNR(dB)")
        plt.savefig("snr_noise.png")
        plt.show()
        """
        # High Period of Noise for some time
        t, c = carrier(amp, fs, N)
        x_n = c

        n_dat = make_data(x_n, Layer_1)

        p_pred = sess.run([pred], feed_dict={x: n_dat})[0]
        l2norm = np.sum(np.square(p_pred - n_dat), 1)



        """
        # High Period of Noise for some time
        noise, _, _ = noise_band(1, 0.01, fs, N)
        x_n = c + noise

        f_n, t_n, Sxx_n = signal.spectrogram(x_n, fs, window='blackmanharris', nperseg=63, noverlap=62)
        #n_dat = Sxx_n.T
        n_dat = make_data(x_n, window)

        p_pred = sess.run([pred], feed_dict={x: n_dat})[0]
        l2norm = np.sum(np.square(p_pred - n_dat), 1)

        plot_spectrum(t, f, p_pred, n_dat, l2norm, "noise_band.png")

        # Rogue Signal for sometime
        noise, _, _ = noise_rogue(1, amp, fs, N)
        x_n = c + noise

        f_n, t_n, Sxx_n = signal.spectrogram(x_n, fs, window='blackmanharris', nperseg=63, noverlap=62)
        #n_dat = Sxx_n.T
        n_dat = make_data(x_n, window)

        p_pred = sess.run([pred], feed_dict={x: n_dat})[0]
        l2norm = np.sum(np.square(p_pred - n_dat), 1)

        plot_spectrum(t, f, p_pred, n_dat, l2norm, "noise_rogue.png")

        step_rate = np.array(step_rate)
        loss_rate = np.array(loss_rate)

        plt.plot(step_rate, loss_rate)
        plt.ylim((0, 0.00002))
        plt.xlabel("Time")
        plt.ylabel("Loss")
        plt.margins(x=0,y=0)
        plt.savefig("learn_rate.png")
        plt.show()
        """
