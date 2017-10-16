import numpy as np
from scipy import signal
import scipy
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
from matplotlib import cm, colors

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
    
        plt.savefig(filename, format='png')
        plt.show()

def calc_snr(noise, signal):
            # Calculate SNR
            p_noise = np.sqrt(1/len(noise)*np.sum(np.square(np.abs(noise))))
            p_sn = np.sqrt(1/len(signal)*np.sum(np.square(np.abs(signal))))
            return 10*np.log10((p_sn)/p_noise)

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

def  make_data(data, window_size):
        X = []

        win_size = int(window_size)
        for i in range(len(data)-win_size):
                win = data[i:i+win_size]
                X.append(win)

        return np.array(X)
