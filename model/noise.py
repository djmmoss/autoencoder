import numpy as np

def noise_band(n_l, amp, siz=0.05, fs=5e3, N=1e4, loc_s=0.0, loc_e=1.0):
        time = np.arange(N) / float(fs)
        noise_power = amp * fs / 2
        noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        i = n_l * np.cos(noise)
        q = n_l * np.sin(noise)
        
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
        mod = n_l*amp*np.exp(2j*np.pi*fs*time)
        i = np.real(mod)
        q = np.imag(mod)
        
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
