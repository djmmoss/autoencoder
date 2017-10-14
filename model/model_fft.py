import numpy as np
import tensorflow as tf
import sys
from scipy import signal
import scipy
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

def noise_band(n_l, amp, siz=0.05, fs=5e3, N=1e4):
        time = np.arange(N) / float(fs)
        noise_power = n_l * amp * fs / 2
        noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        i = amp*np.cos(noise)
        q = amp*np.sin(noise)
        
        s1 = np.random.uniform(0,1,1)
        while (s1 + siz) > 1.0:
            s1 = np.random.rand() 
        n_start = int(len(time)*s1)
        n_end = int(len(time)*(s1+siz))

        i[:n_start] *= 0
        i[n_end:] *= 0
        q[:n_start] *= 0
        q[n_end:] *= 0
        return (i, q, n_start, n_end)

def noise_complex_sine(n_l, amp, siz=0.05, fs=5e3, N=1e4):
        time = np.arange(N) / float(fs)
        fs = np.random.uniform(fs/2, fs*2, 1) 
        mod = amp*np.exp(2j*np.pi*fs*time)
        i = n_l*np.real(mod)
        q = n_l*np.imag(mod)
        
        s1 = np.random.uniform(0,1,1)
        while (s1 + siz) > 1.0:
            s1 = np.random.rand() 
        n_start = int(len(time)*s1)
        n_end = int(len(time)*(s1+siz))

        i[:n_start] *= 0
        i[n_end:] *= 0
        q[:n_start] *= 0
        q[n_end:] *= 0
        return (i, q, n_start, n_end)

def noise_chirp(n_l, amp, siz=0.05, fs=5e3, N=1e4):
        time = np.arange(N) / float(fs)
        
        s1 = np.random.uniform(0,1,1)
        while (s1 + siz) > 1.0:
            s1 = np.random.rand() 
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

def make_fft_windows(sig, window_size):
        sig_window = make_data(sig, window_size)

        data = []
        for sig_w in sig_window:
            fft_res = np.fft.fft(sig_w)
            fft_r = np.real(fft_res)
            fft_i = np.imag(fft_res)
            data.append(np.concatenate((fft_r, fft_i)))

        return np.array(data)


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
N = 1e4
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
        for i in range(20000):
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
        
        t = np.arange(N) / float(fs)
        si, sq = carrier(amp, fs, N)

        noise_level = np.linspace(-20, 20, 20)
        
        c_snr = np.zeros((20, 5))
        c_l2n = np.zeros((20, 5))
        for j in range(5):
            for i in range(20):
                snr = 0.0
                while((np.abs(snr - noise_level[i]) > 1) or np.isnan(snr)):
                    noise = np.random.uniform(0.00001, 10, 1) + 0.0000001
                    i_n, q_n, n_start, n_end = noise_complex_sine(noise, amp, 0.2, fs, N)
                    i_s = si + i_n
                    q_s = sq + q_n
                    
                    s = i_s*np.sin(2*np.pi*2e3*t) + q_s*np.cos(2*np.pi*2e3*t) 
                    s_n = i_n*np.sin(2*np.pi*2e3*t) + q_n*np.cos(2*np.pi*2e3*t) 
                    snr = calc_snr(s_n[n_start:n_end], s[n_start:n_end])

                sig = i_s + q_s*1j
                data = make_fft_windows(sig, int(Layer_1/2))
                p_pred = sess.run([pred], feed_dict={x: data})[0]
                l2norm = np.sum(np.square(p_pred - data),1)
                c_snr[i,j] = snr
                c_l2n[i,j] = np.mean(l2norm[n_start:n_end])/baseline
        
        ch_snr = np.zeros((20, 5))
        ch_l2n = np.zeros((20, 5))
        for j in range(5):
            for i in range(20):
                snr = 0.0
                while((np.abs(snr - noise_level[i]) > 1) or np.isnan(snr)):
                    noise = np.random.uniform(0.00001, 10, 1) + 0.0000001
                    i_n, q_n, n_start, n_end = noise_chirp(noise, amp, 0.2, fs, N)
                    i_s = si + i_n
                    q_s = sq + q_n
                    
                    s = i_s*np.sin(2*np.pi*2e3*t) + q_s*np.cos(2*np.pi*2e3*t) 
                    s_n = i_n*np.sin(2*np.pi*2e3*t) + q_n*np.cos(2*np.pi*2e3*t) 
                    snr = calc_snr(s_n[n_start:n_end], s[n_start:n_end])

                print(snr)
                sig = i_s + q_s*1j
                data = make_fft_windows(sig, int(Layer_1/2))
                p_pred = sess.run([pred], feed_dict={x: data})[0]
                l2norm = np.sum(np.square(p_pred - data),1)
                ch_snr[i,j] = snr
                ch_l2n[i,j] = np.mean(l2norm[n_start:n_end])/baseline
        

        c_snr = np.mean(c_snr, axis=1)
        c_l2n = np.mean(c_l2n, axis=1)
        ch_snr = np.mean(ch_snr, axis=1)
        ch_l2n = np.mean(ch_l2n, axis=1)
        print(c_snr)
        print(c_l2n)
        print(ch_snr)
        print(ch_l2n)
        
        plt.semilogx(c_l2n, c_snr, label="Complex Sinusoid")
        plt.semilogx(ch_l2n, ch_snr, label="Chirp Event")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)
        plt.ylim((-20, 20))
        plt.margins(x=0,y=0)
        plt.xlabel("Number of Times larger than the Basline")
        plt.ylabel("SNR(dB)")
        plt.savefig("snr_noise.png")
        plt.show()
      
        diff = int(Layer_1/2)
        i_s = i_s[:-diff]
        q_s = q_s[:-diff]
        t = np.arange(N) / float(fs)
        t = t[:-diff]
        c = i_s*np.sin(2*np.pi*2e3*t) + q_s*np.cos(2*np.pi*2e3*t) 
        f_sp, t_sp, Sxx = signal.spectrogram(c, fs, window='blackmanharris')
       
        f = np.linspace(0, fs, 32)

        fig = figure(1)

        ax1 = fig.add_subplot(211)
        ax1.pcolormesh(t, f, data.T)
        #ax1.plot(t, c)
        #ax1.margins(x=0,y=0)
        
        ax2 = fig.add_subplot(212)
        #ax2.pcolormesh(t, f, data.T)
        ax2.pcolormesh(t_sp, f_sp, Sxx)

        #ax2 = fig.add_subplot(313)
        #ax2.plot(t, p)
        #ax2.margins(x=0,y=0)
        
        #show()
        
        i_s, q_s = carrier(amp, fs, N)
        i_n, q_n, _, _ = noise_complex_sine(1, amp, 0.05, fs, N)
        i_s = i_s + i_n
        q_s = q_s + q_n
        
        i_n, q_n, _, _ = noise_chirp(1, amp, 0.05, fs, N)
        i_s = i_s + i_n
        q_s = q_s + q_n
        
        i_n, q_n, _, _ = noise_band(1, amp, 0.05, fs, N)
        i_s = i_s + i_n
        q_s = q_s + q_n


        sig = i_s + q_s*1j

        data = make_fft_windows(sig, int(Layer_1/2))
        p_pred = sess.run([pred], feed_dict={x: data})[0]
        l2norm = np.sum(np.square(p_pred - data), 1)
        
        diff = int(Layer_1/2)
        i_s = i_s[:-diff]
        q_s = q_s[:-diff]
        t = np.arange(N) / float(fs)
        t = t[:-diff]
        c = i_s*np.sin(2*np.pi*2e3*t) + q_s*np.cos(2*np.pi*2e3*t) 
        f_sp, t_sp, Sxx = signal.spectrogram(c, fs, window='blackmanharris')
        
        fig = figure(1)

        ax1 = fig.add_subplot(311)
        ax1.pcolormesh(t_sp, f_sp, Sxx)
        #ax1.plot(t, c)
        #ax1.margins(x=0,y=0)
        
        ax2 = fig.add_subplot(312)
        ax2.pcolormesh(t, f, data.T)

        #ax3 = fig.add_subplot(413)
        #ax3.plot(t, p)
        #ax3.margins(x=0,y=0)
        
        ax4 = fig.add_subplot(313)
        ax4.plot(t, l2norm)
        ax4.margins(x=0,y=0)
        
        fig.savefig("noise_overview.png")
        #show()

        i_s, q_s = carrier(amp, fs, N)
        i_n, q_n, _, _ = noise_complex_sine(1, amp, 0.2, fs, N)

        i_s = i_s + i_n
        q_s = q_s + q_n

        sig = i_s + q_s*1j

        data = make_fft_windows(sig, int(Layer_1/2))
        p_pred = sess.run([pred], feed_dict={x: data})[0]
        l2norm = np.sum(np.square(p_pred - data), 1)
        
        diff = int(Layer_1/2)
        i_s = i_s[:-diff]
        q_s = q_s[:-diff]
        t = np.arange(N) / float(fs)
        t = t[:-diff]
        c = i_s*np.sin(2*np.pi*2e3*t) + q_s*np.cos(2*np.pi*2e3*t) 
        f_sp, t_sp, Sxx = signal.spectrogram(c, fs, window='blackmanharris')
        
        fig = figure(1)

        ax1 = fig.add_subplot(311)
        ax1.pcolormesh(t_sp, f_sp, Sxx)
        #ax1.plot(t, c)
        #ax1.margins(x=0,y=0)
        
        ax2 = fig.add_subplot(312)
        ax2.pcolormesh(t, f, data.T)

        #ax3 = fig.add_subplot(413)
        #ax3.plot(t, p)
        #ax3.margins(x=0,y=0)
        
        ax4 = fig.add_subplot(313)
        ax4.plot(t, l2norm)
        ax4.margins(x=0,y=0)
        
        fig.savefig("noise_complex.png")
        #show()
        
        i_s, q_s = carrier(amp, fs, N)
        i_n, q_n, _, _ = noise_chirp(1, amp, 0.2, fs, N)

        i_s = i_s + i_n
        q_s = q_s + q_n

        sig = i_s + q_s*1j

        data = make_fft_windows(sig, int(Layer_1/2))
        p_pred = sess.run([pred], feed_dict={x: data})[0]
        l2norm = np.sum(np.square(p_pred - data), 1)
        
        diff = int(Layer_1/2)
        i_s = i_s[:-diff]
        q_s = q_s[:-diff]
        t = np.arange(N) / float(fs)
        t = t[:-diff]
        c = i_s*np.sin(2*np.pi*2e3*t) + q_s*np.cos(2*np.pi*2e3*t) 
        f_sp, t_sp, Sxx = signal.spectrogram(c, fs, window='blackmanharris')
        
        #i_p = []
        #q_p = []
        #for win in p_pred:
        #    ifft_w = win[:diff] + win[diff:]*1j
        #    ifft_res = np.fft.ifft(ifft_w)
        #    i_p.append(np.real(ifft_res[0]))
        #    q_p.append(np.imag(ifft_res[0]))

        #i_p = np.array(i_p)
        #q_p = np.array(q_p)
        #p = i_p*np.sin(2*np.pi*2e3*t) + q_p*np.cos(2*np.pi*2e3*t) 
        
        fig = figure(1)

        ax1 = fig.add_subplot(311)
        ax1.pcolormesh(t_sp, f_sp, Sxx)
        #ax1.plot(t, c)
        #ax1.margins(x=0,y=0)
        
        ax2 = fig.add_subplot(312)
        ax2.pcolormesh(t, f, data.T)

        #ax3 = fig.add_subplot(413)
        #ax3.plot(t, p)
        #ax3.margins(x=0,y=0)
        
        ax4 = fig.add_subplot(313)
        ax4.plot(t, l2norm)
        ax4.margins(x=0,y=0)
        
        fig.savefig("noise_chirp.png")
        #show()
        
        i_s, q_s = carrier(amp, fs, N)
        i_n, q_n, _, _ = noise_band(1, amp, 0.2, fs, N)

        i_s = i_s + i_n
        q_s = q_s + q_n

        sig = i_s + q_s*1j

        data = make_fft_windows(sig, int(Layer_1/2))
        p_pred = sess.run([pred], feed_dict={x: data})[0]
        l2norm = np.sum(np.square(p_pred - data), 1)
        
        diff = int(Layer_1/2)
        i_s = i_s[:-diff]
        q_s = q_s[:-diff]
        t = np.arange(N) / float(fs)
        t = t[:-diff]
        c = i_s*np.sin(2*np.pi*2e3*t) + q_s*np.cos(2*np.pi*2e3*t) 
        f_sp, t_sp, Sxx = signal.spectrogram(c, fs, window='blackmanharris')
        
        #i_p = []
        #q_p = []
        #for win in p_pred:
        #    ifft_w = win[:diff] + win[diff:]*1j
        #    ifft_res = np.fft.ifft(ifft_w)
        #    i_p.append(np.real(ifft_res[0]))
        #    q_p.append(np.imag(ifft_res[0]))

        #i_p = np.array(i_p)
        #q_p = np.array(q_p)
        #p = i_p*np.sin(2*np.pi*2e3*t) + q_p*np.cos(2*np.pi*2e3*t) 
        
        fig = figure(1)

        ax1 = fig.add_subplot(311)
        ax1.pcolormesh(t_sp, f_sp, Sxx)
        #ax1.plot(t, c)
        #ax1.margins(x=0,y=0)
        
        ax2 = fig.add_subplot(312)
        ax2.pcolormesh(t, f, data.T)

        #ax3 = fig.add_subplot(413)
        #ax3.plot(t, p)
        #ax3.margins(x=0,y=0)
        
        ax4 = fig.add_subplot(313)
        ax4.plot(t, l2norm)
        ax4.margins(x=0,y=0)
        
        fig.savefig("noise_band.png")
        #show()



        """
        noise_level = np.linspace(0.00001, 0.002, 100)
        noise_level = np.concatenate((noise_level, np.linspace(0.002, 0.2, 100)))
        gb_f_snr = []
        gb_f_l2n = []

        for n_l in noise_level:
            noise, n_start, n_end = noise_band(n_l, 0.001, fs, N)
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
