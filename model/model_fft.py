import numpy as np
import tensorflow as tf
import sys
from tqdm import *
from scipy import signal
import scipy
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from tensorflow.examples.tutorials.mnist import input_data

def noise_band(n_l, amp, siz=0.05, fs=5e3, N=1e4, loc_s=0.0, loc_e=1.0):
        time = np.arange(N) / float(fs)
        noise_power = n_l * amp * fs / 2
        noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        i = amp*np.cos(noise)
        q = amp*np.sin(noise)
        
        s1 = np.random.uniform(loc_s,loc_e,1)
        while (s1 + siz) > 1.0:
            s1 = np.random.uniform(loc_s,loc_e,1)
        n_start = int(len(time)*s1)
        n_end = int(len(time)*(s1+siz))

        i[:n_start] *= 0
        i[n_end:] *= 0
        q[:n_start] *= 0
        q[n_end:] *= 0
        return (i, q, n_start, n_end)

def noise_complex_sine(n_l, amp, siz=0.05, fs=5e3, N=1e4, loc_s=0.0, loc_e=1.0):
        time = np.arange(N) / float(fs)
        fs = np.random.uniform(fs/2, fs*2, 1) 
        mod = amp*np.exp(2j*np.pi*fs*time)
        i = n_l*np.real(mod)
        q = n_l*np.imag(mod)
        
        s1 = np.random.uniform(loc_s,loc_e,1)
        while (s1 + siz) > 1.0:
            s1 = np.random.uniform(loc_s,loc_e,1)
        n_start = int(len(time)*s1)
        n_end = int(len(time)*(s1+siz))

        i[:n_start] *= 0
        i[n_end:] *= 0
        q[:n_start] *= 0
        q[n_end:] *= 0
        return (i, q, n_start, n_end)

def noise_chirp(n_l, amp, siz=0.05, fs=5e3, N=1e4, loc_s=0.0, loc_e=1.0):
        time = np.arange(N) / float(fs)
        
        s1 = np.random.uniform(loc_s,loc_e,1)
        while (s1 + siz) > 1.0:
            s1 = np.random.uniform(loc_s,loc_e,1)
        n_start = int(len(time)*s1)
        n_end = int(len(time)*(s1+siz))
        
        length = int(len(time[n_start:n_end]))
        fs1 = np.random.uniform(fs/2, fs*2, 1) 
        fs2 = np.random.uniform(fs/2, fs*2, 1) 
        freq = np.linspace(fs1, fs2, length)
        freq_start = np.ones(len(time[:n_start]))*fs1
        freq_end = np.ones(len(time[n_end:]))*fs2
        freq = np.concatenate((freq_start, freq))
        freq = np.concatenate((freq, freq_end))
        mod = n_l*amp*np.exp(2j*np.pi*freq*time)
        i = np.real(mod)
        q = np.imag(mod)
        i[:n_start] *= 0
        i[n_end:] *= 0
        q[:n_start] *= 0
        q[n_end:] *= 0
        return (i, q, n_start, n_end)

def make_windows(i, q, window_size, do_fft=True):
    if do_fft:
        return make_fft_windows(i, q, window_size)
    else:
        return make_iq_windows(i, q, window_size)

def make_iq_windows(i, q, window_size):
    i_data = make_data(i_s, window_size)
    q_data = make_data(q_s, window_size)
    return np.concatenate((i_data, q_data), axis=1)


def make_fft_windows(i, q, window_size):
        sig = i + q*1j
        sig_window = make_data(sig, window_size)

        data = []
        for sig_w in sig_window:
            fft_res = np.fft.fft(sig_w)
            fft_r = np.real(fft_res)
            fft_i = np.imag(fft_res)
            data.append(np.concatenate((fft_r, fft_i)))

        return np.array(data)


def plot_results(t, f, s, data, l2norm, filename, do_fft=True):
       
        fontsize = 12
        plt.close('all')
        fig = plt.figure()

        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
       
        if do_fft:
            norm = colors.Normalize(vmin = np.nanmin(data), vmax = np.nanmax(data))
            ax1.pcolormesh(t, f, data.T, norm=norm, cmap=cm.inferno)
            #ax1.set_xlabel('Time', fontsize=fontsize)
            ax1.set_ylabel('Frequency', fontsize=fontsize)
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            ax1.plot(t, s)
            ax1.set_ylabel('Amplitude', fontsize=fontsize)
            ax1.margins(x=0,y=0)


        ax2.plot(t, l2norm)
        ax2.axhline(y=np.mean(l2norm), color='r', linestyle='-')
        ax2.margins(x=0,y=0)
        ax2.set_xlabel('Time', fontsize=fontsize)
        ax2.set_ylabel('L2-Norm', fontsize=fontsize)
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()   
    
        plt.savefig(filename, format='pdf')
        plt.show()

def calc_snr(noise, signal):
            # Calculate SNR
            p_noise = 1/len(noise)*np.sum(np.square(np.abs(noise)))
            p_sn = 1/len(signal)*np.sum(np.square(np.abs(signal)))
            return 10*np.log10((p_sn - p_noise)/p_noise)

def carrier(amp,fs=5e3, N=1e4):
        time = np.arange(N) / float(fs)
        noise = np.random.normal(scale=0.001, size=time.shape) 
        mod_2 = 0.5*np.cos(2*np.pi*25*time)
        mod = 25*np.cos(2*np.pi*5*time + mod_2)
        #mod = 500*np.cos(2*np.pi*0.25*time)# + noise
        i = amp*np.cos(mod)
        q = amp*np.sin(mod)
        #c_test = amp * np.sin(2*np.pi*2e3*time + mod)
        #c = i*np.sin(2*np.pi*2e3*time) + q*np.cos(2*np.pi*2e3*time) 
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
        f.write("{ .re = " + str(np.real(val)) + ", .im = " + str(np.imag(val)) + "},\n")

    val = np.exp(2j*np.pi*L1*(L1/L1))
    f.write("{ .re = " + str(np.real(val)) + ", .im = " + str(np.imag(val)) + "}};")
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

fs = 5e3
N = 1e3
amp = 2 * np.sqrt(2)
do_fft = False 

i_s, q_s = carrier(amp, fs, N)

data = make_windows(i_s, q_s, int(Layer_1/2), do_fft)

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
        for i in range(30000):
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
        np.savetxt(data_path + 'data.out', x_test[:,0], delimiter=',')
        np.savetxt(data_path + 'expected.out', l2norm, delimiter=',')
        print_weights(data_path, n_weights)
        print_biases(data_path, n_biases)
        print_network(Layer_1, Layer_2, Layer_3)
        print_fft_w(int(Layer_1/2))

        p_pred = sess.run([pred], feed_dict={x: data})[0]
        baseline = np.mean(np.sum(np.square(p_pred - data), 1))
      
        diff = int(Layer_1/2)
        t = np.arange(N) / float(fs)
        t = t[:-diff]
        f = np.linspace(0, fs, 32)
        
        i_s, q_s = carrier(amp, fs, N)
        i_n, q_n, _, _ = noise_complex_sine(1, amp, 0.05, fs, N, 0, 0.2)
        i_s = i_s + i_n
        q_s = q_s + q_n
        
        i_n, q_n, _, _ = noise_chirp(1, amp, 0.05, fs, N, 0.25, 0.5)
        i_s = i_s + i_n
        q_s = q_s + q_n
        
        i_n, q_n, _, _ = noise_band(1, amp, 0.05, fs, N, 0.55, 1.0)
        i_s = i_s + i_n
        q_s = q_s + q_n
        

        data = make_windows(i_s, q_s, int(Layer_1/2), do_fft)
        p_pred = sess.run([pred], feed_dict={x: data})[0]
        l2norm = np.sum(np.square(p_pred - data), 1)
        
        i_s = i_s[:-diff]
        q_s = q_s[:-diff]
        c = i_s*np.cos(2*np.pi*2e3*t) - q_s*np.sin(2*np.pi*2e3*t) 
        
        plot_results(t, f, c, data, l2norm, "noise_overview.pdf", do_fft)

        t = np.arange(N) / float(fs)
        si, sq = carrier(amp, fs, N)

        noise_level = np.linspace(-20, 20, 20)
       
        tests = 10

        c_snr = np.zeros((20, tests))
        c_l2n = np.zeros((20, tests))
        c_false = np.zeros((20, tests))
        for j in range(tests):
            for i in tqdm(range(20)):
                snr = 0.0
                while((np.abs(snr - noise_level[i]) > 1) or np.isnan(snr)):
                    noise = np.random.uniform(0.00001, 10, 1)
                    i_n, q_n, n_start, n_end = noise_complex_sine(noise, amp, 0.2, fs, N)
                    i_s = si + i_n
                    q_s = sq + q_n
                    
                    s = i_s*np.sin(2*np.pi*2e3*t) + q_s*np.cos(2*np.pi*2e3*t) 
                    s_n = i_n*np.sin(2*np.pi*2e3*t) + q_n*np.cos(2*np.pi*2e3*t) 
                    snr = calc_snr(s_n[n_start:n_end], s[n_start:n_end])

                data = make_windows(i_s, q_s, int(Layer_1/2), do_fft)
                p_pred = sess.run([pred], feed_dict={x: data})[0]
                l2norm = np.sum(np.square(p_pred - data),1)
                c_snr[i,j] = snr
                c_l2n[i,j] = np.mean(l2norm[n_start:n_end])/np.mean(l2norm) > 1
                l2norm[n_start:n_end] = 0
                num_false = l2norm > np.mean(l2norm)
                c_false[i,j] = np.sum(num_false)/len(num_false)
        
        ch_snr = np.zeros((20, tests))
        ch_l2n = np.zeros((20, tests))
        ch_false = np.zeros((20, tests))
        for j in range(tests):
            for i in tqdm(range(20)):
                snr = 0.0
                while((np.abs(snr - noise_level[i]) > 1) or np.isnan(snr)):
                    noise = np.random.uniform(0.00001, 10, 1)
                    i_n, q_n, n_start, n_end = noise_chirp(noise, amp, 0.2, fs, N)
                    i_s = si + i_n
                    q_s = sq + q_n
                    
                    s = i_s*np.sin(2*np.pi*2e3*t) + q_s*np.cos(2*np.pi*2e3*t) 
                    s_n = i_n*np.sin(2*np.pi*2e3*t) + q_n*np.cos(2*np.pi*2e3*t) 
                    snr = calc_snr(s_n[n_start:n_end], s[n_start:n_end])

                data = make_windows(i_s, q_s, int(Layer_1/2), do_fft)
                p_pred = sess.run([pred], feed_dict={x: data})[0]
                l2norm = np.sum(np.square(p_pred - data),1)
                ch_snr[i,j] = snr
                ch_l2n[i,j] = np.mean(l2norm[n_start:n_end])/np.mean(l2norm) > 1
                l2norm[n_start:n_end] = 0
                num_false = l2norm > np.mean(l2norm)
                ch_false[i,j] = np.sum(num_false)/len(num_false)
        

        c_snr = np.mean(c_snr, axis=1)
        c_l2n = np.sum(c_l2n, axis=1)/tests
        c_false = np.mean(c_false, axis=1)
        ch_snr = np.mean(ch_snr, axis=1)
        ch_l2n = np.sum(ch_l2n, axis=1)/tests
        ch_false = np.mean(ch_false, axis=1)
        
        plt.close('all')
        plt.plot(c_snr, c_l2n, label="Complex Sinusoid")
        plt.plot(c_snr, c_false, label="Complex Sinusoid (False)")
        plt.plot(ch_snr, ch_l2n, label="Chirp Event")
        plt.plot(ch_snr, ch_false, label="Chirp Event (False)")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)
        plt.margins(x=0,y=0)
        plt.ylim(0,1.05)
        plt.xlim(-20,20)
        plt.xlabel("SNR(dB)")
        plt.ylabel("Probability of Detection")
        plt.savefig("snr_noise.pdf", format="pdf")
        show()
