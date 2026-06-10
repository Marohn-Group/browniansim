# given 2D array of cantilever signal (voltage) vs. time
# demodulate the signal using freqdemod functions and return the cantilever frequency (fc)
import numpy as np
import math

# demodulation functions
def time_mask_binarate(x, y, dt, mode):

    n = y.shape[0] # numer of points in the signal
    indices = np.arange(n) # np.array of indices

    # nearest power of 2 to n
    n2 = int(math.pow(2, int(math.floor(math.log(n,2)))))

    if mode == "middle":
        n_start = int(math.floor((n - n2)/2))
        n_stop = int(n_start + n2)

    elif mode == "start":
        n_start = 0 
        n_stop = n2

    elif mode == "end":
        n_start = n-n2
        n_stop = n

    mask = (indices >= n_start) & (indices < n_stop)

    x = x[mask]
    y = y[mask]

    return x, y

def time_window_cyclicize(x, y, dt, tw):

    n = y.shape[0]
    ww = int(math.ceil((1.0*tw)/(1.0*dt)))
    tw_actual = ww*dt
    w = np.concatenate([np.blackman(2*ww)[0:ww], np.ones(n-2*ww), np.blackman(2*ww)[-ww:]])
    
    return w

def fft(x, y, dt, psd=False):
    
    freq = np.fft.fftshift(np.fft.fftfreq(y.shape[0], dt))

    if psd == False:
        sFT = dt * np.fft.fftshift(np.fft.fft(y))

    elif psd == True:
        sFT = (dt / len(y)) * np.power(abs(np.fft.fftshift(np.fft.fft(y))), 2.0)
        # single-sided power s
        mask = freq >= 0
        freq = freq[mask]
        sFT = sFT[mask]

    freq = freq/1E3 # kHz

    return freq, sFT

def freq_filter_Hilbert_complex(freq, sFT):

    filt = 0.0*(freq < 0) + 1.0*(freq == 0) + 2.0*(freq > 0)

    return filt

def freq_filter_bp(freq, sFT, filt, bw, order=50, style="brick wall"):

    # center freq, fc, is peak in abs of FT spectrum
    FTrh = filt*abs(sFT)
    fc = freq[np.argmax(FTrh)]

    # compute the filter
    freq_scaled = (freq - fc)/bw

    if style == "brick wall":
        bp = 1.0/(1.0+np.power(abs(freq_scaled), order))

    elif style == "cosine":
        # use a trick
        bp = np.zeros(freq.shape)
        sub_index = (freq >= -1.0*bw + fc) & (freq <= bw + fc)
        sub_indices = np.arange(freq.size)[sub_index]
        bp[sub_indices] = np.sin(np.linspace(0,np.pi,sub_indices.size))

    elif style == "gaussian":

            bp = np.exp(-1 * np.power(freq_scaled, 2.0))

    else:

        print("**ERROR**: Unrecognized filter function")

    return bp

def time_mask_rippleless(x, dt, td):

    ww = int(math.ceil((1.0*td)/(1.0*dt)))
    td_actual = ww*dt

    n = x.size
    indices = np.arange(n)
    mask = (indices >= ww) & (indices < n - ww)
    x_rippleless = x[mask]

    return mask, x_rippleless

def ifft(sFT, dt, filt, bp, x_rippleless):

    # divide the FT-ed data by the timestep to recover the digital Fourier transformed data
    # Carry out hte transforms

    s = sFT / dt

    s = s*filt
    s = s*bp 

    sIFT = np.fft.ifft(np.fft.ifftshift(s))

    mask = np.array(x_rippleless)
    sIFT = sIFT[mask]

    p = np.unwrap(np.angle(sIFT))/(2*np.pi)

    a = abs(sIFT)

    return sIFT, p, a

def fit_phase(sIFT, p, a, dt, x, dt_chunk_target):

    n = p.size # no. of phase points
    n_per_chunk = int(round(dt_chunk_target/dt)) # points per chunk
    dt_chunk = dt*n_per_chunk # actual time per chunk
    n_tot_chunk = int(n/n_per_chunk) # total number of chunks
    n_total = n_per_chunk*n_tot_chunk # (realizable) no. of phase points

    # reshape the phase data and
    # zero the phase at the start of each chunk

    y = np.array(p[0:n_total])
    y_sub = y.reshape((n_tot_chunk,n_per_chunk))
    y_sub_reset = y_sub - y_sub[:,:,np.newaxis][:,0,:]*np.ones(n_per_chunk)

    # reshape the time data and
    # zero the time at start of each chunk

    x = np.array(x[0:n_total])
    x_sub = x.reshape((n_tot_chunk, n_per_chunk))
    x_sub_reset = x_sub - x_sub[:,:,np.newaxis][:,0,:]*np.ones(n_per_chunk)

    # use linear least-squares fitting formulas
    #  to calculate the best-fit slope

    SX = dt*0.50*(n_per_chunk-1)*(n_per_chunk)
    SXX = (dt)**2*(1/6.0)*(n_per_chunk)*(n_per_chunk-1)*(2*n_per_chunk-1)
    SY = np.sum(y_sub_reset,axis=1)
    SXY = np.sum(x_sub_reset*y_sub_reset,axis=1)
    slope = (n_per_chunk*SXY-SX*SY)/(n_per_chunk*SXX-SX*SX)

    x_sub_middle = np.mean(x_sub[:,:],axis=1)

    return x_sub_middle, slope

def getCantileverFreq(data,dt_chunk_target = 221.34E-6):

    # create numpy array from incoming waveform
    arr = np.array(data)
    # extract ordinate data array
    x, y = arr[:,0], arr[:,1]
    # time step
    dt = x[1] - x[0]  
    # apply time mask
    # x, y = time_mask_binarate(x, y, dt, "middle")
    # force data to start/end at zero, prevent edge effect on the beginning and end of the signal
    w = time_window_cyclicize(x, y, dt, 3E-3)
    # apply cyclicizing window to signal
    y = y*w
    # fourier transform (F.T.)
    freq, sFT = fft(x, y, dt)
    # complex Hilbert transform
    filt = freq_filter_Hilbert_complex(freq, sFT)
    # apply 1kHz wide bandpass filter
    bp = freq_filter_bp(freq, sFT, filt, 1.00)
    # set up filter to remove ripples
    mask, x_rippleless = time_mask_rippleless(x, dt, 15E-3)
    # Inverse F.T.
    sIFT, p, a = ifft(sFT, dt, filt, bp, mask)
    # fit phase vs. time data
    x_sub_middle, slope = fit_phase(sIFT, p, a, dt, x, dt_chunk_target)
    # compute average fc
    avg_fc = np.mean(slope)

    return slope

def getCantileverFreq_noWindow(data):

    # create numpy array from incoming waveform
    arr = np.array(data)
    # extract ordinate data array
    x, y = arr[:,0], arr[:,1]
    # time step
    dt = x[1] - x[0]  
    # apply time mask
    # x, y = time_mask_binarate(x, y, dt, "middle")
    # force data to start/end at zero, prevent edge effect on the beginning and end of the signal
    # w = time_window_cyclicize(x, y, dt, 3E-3)
    # # apply cyclicizing window to signal
    # y = y*w
    # fourier transform (F.T.)
    freq, sFT = fft(x, y, dt)
    # complex Hilbert transform
    filt = freq_filter_Hilbert_complex(freq, sFT)
    # apply 1kHz wide bandpass filter
    bp = freq_filter_bp(freq, sFT, filt, 1.00)
    # set up filter to remove ripples
    # mask, x_rippleless = time_mask_rippleless(x, dt, 15E-3)
    # Inverse F.T.
    sIFT, p, a = ifft(sFT, dt, filt, bp, np.ones(x.size))
    # fit phase vs. time data
    x_sub_middle, slope = fit_phase(sIFT, p, a, dt, x, 221.34E-6)
    # compute average fc
    avg_fc = np.mean(slope)

    return slope



def getCantileverPhase(data):

    # create numpy array from incoming waveform
    arr = np.array(data)
    # extract ordinate data array
    x, y = arr[:,0], arr[:,1]
    # time step
    dt = x[1] - x[0]  
    # apply time mask
    # x, y = time_mask_binarate(x, y, dt, "middle")
    # force data to start/end at zero, prevent edge effect on the beginning and end of the signal
    w = time_window_cyclicize(x, y, dt, 3E-3)
    # apply cyclicizing window to signal
    y = y*w
    # fourier transform (F.T.)
    freq, sFT = fft(x, y, dt)
    # complex Hilbert transform
    filt = freq_filter_Hilbert_complex(freq, sFT)
    # apply 1kHz wide bandpass filter
    bp = freq_filter_bp(freq, sFT, filt, 1.00)
    # set up filter to remove ripples
    mask, x_rippleless = time_mask_rippleless(x, dt, 1E-3)
    # Inverse F.T.
    sIFT, p, a = ifft(sFT, dt, filt, bp, mask)
    # fit phase vs. time data
    # x_sub_middle, slope = fit_phase(sIFT, p, a, dt, x, 221.34E-6)
    # # compute average fc
    # avg_fc = np.mean(slope)

    return p


def getCantileverComplexTrajectory(data,bw = 1.00):

    # create numpy array from incoming waveform
    arr = np.array(data)
    # extract ordinate data array
    x, y = arr[:,0], arr[:,1]
    # time step
    dt = x[1] - x[0]  
    # apply time mask
    # x, y = time_mask_binarate(x, y, dt, "middle")
    # force data to start/end at zero, prevent edge effect on the beginning and end of the signal
    w = time_window_cyclicize(x, y, dt, 3E-3)
    # apply cyclicizing window to signal
    y = y*w
    # fourier transform (F.T.)
    freq, sFT = fft(x, y, dt)
    # complex Hilbert transform
    filt = freq_filter_Hilbert_complex(freq, sFT)
    # apply 1kHz wide bandpass filter
    bp = freq_filter_bp(freq, sFT, filt, bw)
    # set up filter to remove ripples
    mask, x_rippleless = time_mask_rippleless(x, dt, 15E-3)
    # Inverse F.T.
    sIFT, p, a = ifft(sFT, dt, filt, bp, mask)
    # fit phase vs. time data

    return sIFT