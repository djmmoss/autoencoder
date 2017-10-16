import numpy as np
import tensorflow as tf
import sys
from tqdm import *
import pickle
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from noise import *
from sig_gen import *
from c_gen import *

# * Layer1: 32
# * Layer2: 16
# * Layer3: 8
# * Layer4: 16
# * Layer5: 32
Layer_1 = 32
Layer_2 = 16
Layer_3 = 8

fs = 5e3
N = int(1e3)
amp = 1 
do_fft = True 

# Synthetic Data
i_s, q_s = carrier(amp, fs, N)

# Real Data
#x = scipy.fromfile(open("fm_data.bin"), dtype=scipy.complex64)
#i_s = np.real(x[:N])
#q_s = np.imag(x[:N])

data = make_windows(i_s, q_s, int(Layer_1/2), do_fft)

n_data = data
train_size = int(len(data)*0.8)

x_train = n_data[0:train_size,:]
y_train = data[0:train_size,:]
x_test = n_data[train_size:,:]
y_test = data[train_size:,:]


n_weights = {
        'w_l2' : tf.Variable(tf.truncated_normal([Layer_1,Layer_2], stddev=0.1), name="w_l2"),
        'w_l3' : tf.Variable(tf.truncated_normal([Layer_2,Layer_3], stddev=0.1), name="w_l3"),
        'w_l5' : tf.Variable(tf.truncated_normal([Layer_3,Layer_2], stddev=0.1), name="w_l5"),
        'w_l6' : tf.Variable(tf.truncated_normal([Layer_2,Layer_1], stddev=0.1), name="w_l6"),
}

n_biases = {
        'b_l2' : tf.Variable(tf.constant(0.1, shape=[Layer_2]), name="b_l2"),
        'b_l3' : tf.Variable(tf.constant(0.1, shape=[Layer_3]), name="b_l3"),
        'b_l5' : tf.Variable(tf.constant(0.1, shape=[Layer_2]), name="b_l5"),
        'b_l6' : tf.Variable(tf.constant(0.1, shape=[Layer_1]), name="b_l6"),
}

x = tf.placeholder(tf.float32, [None, Layer_1])

y = tf.placeholder(tf.float32, [None, Layer_1])

# Layer 1
a_l2 = tf.matmul(x, n_weights["w_l2"]) + n_biases["b_l2"]
r_l2 = tf.nn.relu(a_l2)

# Layer 2 
a_l3 = tf.matmul(r_l2, n_weights["w_l3"]) + n_biases["b_l3"]
r_l3 = tf.nn.relu(a_l3)

# Layer 3
a_l5 = tf.matmul(r_l3, n_weights["w_l5"]) + n_biases["b_l5"]
r_l5 = tf.nn.relu(a_l5)

# Layer 4
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
        print("MSE(Test Set): ", np.mean(np.square(p_pred_n - y_test)))
        l2norm = np.sum(np.square(p_pred_n - x_test), 1)

        data_path = "data/"
        np.savetxt(data_path + 'data_i.out', i_s, delimiter=',')
        np.savetxt(data_path + 'data_q.out', q_s, delimiter=',')
        np.savetxt(data_path + 'expected.out', l2norm, delimiter=',')
        print_weights(data_path, n_weights)
        print_biases(data_path, n_biases)
        print_network(data_path, Layer_1, Layer_2, Layer_3)
        print_fft_w(data_path, int(Layer_1/2))

        p_pred = sess.run([pred], feed_dict={x: data})[0]
        baseline = np.mean(np.sum(np.square(p_pred - data), 1))
      
        diff = int(Layer_1/2)
        t = np.arange(N) / float(fs)
        t = t[:-diff]
        f = np.linspace(0, fs, 32)
        
        i_s, q_s = carrier(amp, fs, N)
        i_n, q_n, _, _ = noise_complex_sine(0.5, amp, 0.05, fs, N, 0, 0.2)
        i_s = i_s + i_n
        q_s = q_s + q_n
        
        i_n, q_n, _, _ = noise_chirp(0.5, amp, 0.05, fs, N, 0.25, 0.5)
        i_s = i_s + i_n
        q_s = q_s + q_n
        
        i_n, q_n, _, _ = noise_band(0.5, amp, 0.05, fs, N, 0.55, 1.0)
        i_s = i_s + i_n
        q_s = q_s + q_n
        

        data = make_windows(i_s, q_s, int(Layer_1/2), do_fft)
        p_pred = sess.run([pred], feed_dict={x: data})[0]
        l2norm = np.sum(np.square(p_pred - data), 1)
        
        i_s = i_s[:-diff]
        q_s = q_s[:-diff]
        c = i_s*np.cos(2*np.pi*2e3*t) - q_s*np.sin(2*np.pi*2e3*t) 
       
        if do_fft:
            filename = "results/noise_overview_freq.png"
        else:
            filename = "results/noise_overview_time.png"

        plot_results(t, f, c, data, l2norm, filename, do_fft)

        t = np.arange(N) / float(fs)
        si, sq = carrier(amp, fs, N)

        noise_level = np.linspace(20, -20, 20)

        tests = 40
        c_snr = np.zeros((20, tests))
        c_l2n = np.zeros((20, tests))
        c_false = np.zeros((20, tests))
        for j in tqdm(range(tests)):
            for i in range(20):
                snr = 0.0
                noise = 0.00001
                while((np.abs(snr - noise_level[i]) > 1) or np.isnan(snr)):
                    i_n, q_n, n_start, n_end = noise_complex_sine(noise, amp, 0.2, fs, N)

                    s = si*np.sin(2*np.pi*2e3*t) + sq*np.cos(2*np.pi*2e3*t)
                    s_n = i_n*np.sin(2*np.pi*2e3*t) + q_n*np.cos(2*np.pi*2e3*t)
                    snr = calc_snr(s_n, s)
                    noise *= 1.1
                    
                i_s = si + i_n
                q_s = sq + q_n

                data = make_windows(i_s, q_s, int(Layer_1/2), do_fft)
                p_pred = sess.run([pred], feed_dict={x: data})[0]
                l2norm = np.sum(np.square(p_pred - data),1)
                c_snr[i,j] = snr
                c_l2n[i,j] = np.mean(l2norm[n_start:n_end])/(np.mean(l2norm)*1) > 1
                l2norm[n_start:n_end] = 0
                num_false = l2norm > (np.mean(l2norm)*1)
                c_false[i,j] = np.sum(num_false)/len(num_false)
        
        g_snr = np.zeros((20, tests))
        g_l2n = np.zeros((20, tests))
        g_false = np.zeros((20, tests))
        for j in tqdm(range(tests)):
            for i in range(20):
                snr = 0.0
                noise = 0.00001
                while((np.abs(snr - noise_level[i]) > 1) or np.isnan(snr)):
                    i_n, q_n, n_start, n_end = noise_band(noise, amp, 0.2, fs, N)

                    s = si*np.sin(2*np.pi*2e3*t) + sq*np.cos(2*np.pi*2e3*t)
                    s_n = i_n*np.sin(2*np.pi*2e3*t) + q_n*np.cos(2*np.pi*2e3*t)
                    snr = calc_snr(s_n[n_start:n_end], s[n_start:n_end])
                    noise *= 1.1
                    
                i_s = si + i_n
                q_s = sq + q_n

                data = make_windows(i_s, q_s, int(Layer_1/2), do_fft)
                p_pred = sess.run([pred], feed_dict={x: data})[0]
                l2norm = np.sum(np.square(p_pred - data),1)
                g_snr[i,j] = snr
                g_l2n[i,j] = np.mean(l2norm[n_start:n_end])/(np.mean(l2norm)*1) > 1
                l2norm[n_start:n_end] = 0
                num_false = l2norm > (np.mean(l2norm)*1)
                g_false[i,j] = np.sum(num_false)/len(num_false)
        
        ch_snr = np.zeros((20, tests))
        ch_l2n = np.zeros((20, tests))
        ch_num = np.zeros((20, tests))
        ch_false = np.zeros((20, tests))
        for j in tqdm(range(tests)):
            for i in range(20):
                snr = 0.0
                noise = 0.00001
                while((np.abs(snr - noise_level[i]) > 1) or np.isnan(snr)):
                    i_n, q_n, n_start, n_end = noise_chirp(noise, amp, 0.2, fs, N)

                    s = si*np.sin(2*np.pi*2e3*t) + sq*np.cos(2*np.pi*2e3*t)
                    s_n = i_n*np.sin(2*np.pi*2e3*t) + q_n*np.cos(2*np.pi*2e3*t)
                    snr = calc_snr(s_n[n_start:n_end], s[n_start:n_end])
                    noise *= 1.1
                    
                i_s = si + i_n
                q_s = sq + q_n

                data = make_windows(i_s, q_s, int(Layer_1/2), do_fft)
                p_pred = sess.run([pred], feed_dict={x: data})[0]
                l2norm = np.sum(np.square(p_pred - data),1)
                ch_snr[i,j] = snr
                ch_l2n[i,j] = np.mean(l2norm[n_start:n_end])/(np.mean(l2norm)*1) > 1
                num_pred = l2norm > (np.mean(l2norm*1))
                ch_num[i,j] = np.sum(num_pred)/len(num_pred)
                l2norm[n_start:n_end] = 0
                num_false = l2norm > (np.mean(l2norm*1))
                ch_false[i,j] = np.sum(num_false)/len(num_false)
        
        c_snr = np.mean(c_snr, axis=1)
        c_l2n = np.sum(c_l2n, axis=1)/tests
        c_false = np.mean(c_false, axis=1)
        ch_snr = np.mean(ch_snr, axis=1)
        ch_l2n = np.sum(ch_l2n, axis=1)/tests
        ch_false = np.mean(ch_false, axis=1)
        ch_num = np.mean(ch_num, axis=1)
        g_snr = np.mean(g_snr, axis=1)
        g_l2n = np.sum(g_l2n, axis=1)/tests
        g_false = np.mean(g_false, axis=1)
        

        if do_fft:
            c_freq = c_l2n-c_false
            ch_freq = ch_l2n-ch_false
            g_freq = g_l2n-g_false
            pickle.dump(c_freq, open( "sar/c_freq.p", "wb" ) )
            pickle.dump(ch_freq, open( "sar/ch_freq.p", "wb" ) )
            pickle.dump(g_freq, open( "sar/g_freq.p", "wb" ) )
            c_time = pickle.load( open( "sar/c_time.p", "rb" ) )
            ch_time = pickle.load( open( "sar/ch_time.p", "rb" ) )
            g_time = pickle.load( open( "sar/g_time.p", "rb" ) )
        else:
            c_time = c_l2n-c_false
            ch_time = ch_l2n-ch_false
            g_time = g_l2n-g_false
            pickle.dump(c_time, open( "sar/c_time.p", "wb" ) )
            pickle.dump(ch_time, open( "sar/ch_time.p", "wb" ) )
            pickle.dump(g_time, open( "sar/g_time.p", "wb" ) )
            c_freq = pickle.load( open( "sar/c_freq.p", "rb" ) )
            ch_freq = pickle.load( open( "sar/ch_freq.p", "rb" ) )
            g_freq = pickle.load( open( "sar/g_freq.p", "rb" ) )

        plt.close('all')
        plt.plot(noise_level, c_freq, label="Complex Sinusoid (Freq)")
        plt.plot(noise_level, ch_freq, label="Chirp Event (Freq)")
        plt.plot(noise_level, g_freq, label="Gaussian Band (Freq)")

        plt.plot(noise_level, c_time, label="Complex Sinusoid (Time)")
        plt.plot(noise_level, ch_time, label="Chirp Event (Time)")
        plt.plot(noise_level, g_time, label="Gaussian Band (Time)")
        plt.legend()
        plt.margins(x=0,y=0)
        plt.ylim(0,1.05)
        plt.xlim(-20,20)
        plt.xlabel("Signal-to-Anomaly Ratio (dB)")
        plt.ylabel("Probability of Correct Detection")
        plt.tight_layout()   
        plt.savefig("results/sar_noise.pdf", format="pdf")
        show()
        
