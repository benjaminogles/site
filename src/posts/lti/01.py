
# lit text
#
# Linear and Time-Invariant (LTI) Systems
# =======================================
#
# A system is a process that produces an output signal from an input signal.
# Perhaps the simplest example of an input signal, called an impulse, is shown below.
# The only non-zero sample of an impulse occurs at time zero and has a value of one.
#
# lit skip

import numpy as np
import matplotlib.pyplot as plt

n = 9
x = np.zeros(n)
x[n//2] = 1
idx = np.arange(-(n//2), n//2+1)
plt.stem(idx, x, basefmt=' ')
plt.xticks(idx)
plt.ylim([-.1, 1.1])
plt.title('unit impulse')
plt.ylabel('real sample value')
plt.xlabel('time')
plt.savefig('impulse.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# Suppose we have a system that produces the following output signal from an impulse input signal.
# This is called the system's impulse response.
#
# lit skip

h = np.zeros(n)
h[n//2-1:n//2+2] = 1.0
plt.stem(idx, np.convolve(x, h, mode='same'), basefmt=' ')
plt.xticks(idx)
plt.ylim([-.1, 1.1])
plt.title('unit impulse response')
plt.ylabel('real sample value')
plt.xlabel('time')
plt.savefig('h.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# If the system is linear and time-invariant, then it will produce weighted sums of shifted impulse responses from weighted sums of shifted impulse signals.
#
# lit skip

fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)
axs[0][0].stem(idx, x, basefmt=' ')
axs[0][1].stem(idx, np.convolve(x, h, mode='same'), basefmt=' ')
axs[1][0].stem(idx, -1 * np.roll(x, -1), basefmt=' ')
axs[1][1].stem(idx, np.convolve(-1 * np.roll(x, -1), h, mode='same'), basefmt=' ')
axs[2][0].stem(idx, 2 * np.roll(x, 1), basefmt=' ')
axs[2][1].stem(idx, np.convolve(2 * np.roll(x, 1), h, mode='same'), basefmt=' ')
axs[3][0].stem(idx, x + -1 * np.roll(x, -1) + 2 * np.roll(x, 1), basefmt=' ')
axs[3][1].stem(idx, np.convolve(x + -1 * np.roll(x, -1) + 2 * np.roll(x, 1), h, mode='same'), basefmt=' ')
axs[0][0].set_title('input')
axs[0][1].set_title('output')
axs[0][0].set_xticks(idx)
axs[0][1].set_xticks(idx)
axs[0][0].set_ylim([-1.5, 3.5])
axs[0][1].set_ylim([-1.5, 3.5])
axs[1][0].set_title('+')
axs[1][1].set_title('+')
axs[2][0].set_title('+')
axs[2][1].set_title('+')
axs[3][0].set_title('=')
axs[3][1].set_title('=')
plt.tight_layout()
plt.savefig('lti.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# Because every input signal can be constructed as a weighted sum of shifted impulses, the input/output behavior of an LTI system is completely characterized by its impulse response.
# Every output sample is a weighted sum of a set of input samples with the weights given by the system's input response.
# This is called convolution.
#
# In general, the length of an LTI system impulse response may be infinite (IIR), which complicates digital implementation beyond basic convolution.
# As such, many digital LTI systems are designed to have a finite impulse response (FIR) so they can be directly implemented as convolution with the designed impulse response.
#
# An Example System
# -----------------
#
# The following LTI system smooths an input signal by assigning each output sample the average value of a set of input samples.
#

def mean_filter(x, n):
    # require filter length to be odd and not larger than the length of the input
    assert n & 1 and n <= len(x)
    # weight each entry in the impulse response equally
    h = np.ones(n) / n
    # slide the weights along the input, computing a weighted sum at each point
    return np.convolve(x, h)

# lit text
#
# Another way of understanding this system is to analyze its input/output relationship in terms of frequency.
# In another [post](/posts/dft/), I showed that the DFT of the first column in a circulant matrix gives the weights applied to each frequency component of an input vector to yield the DFT of the convolution result.
# The first column of a circulant matrix implementing an FIR LTI system is the system's impulse response, zero padded to prevent the computation from wrapping around at the edges (circular convolution).
# So we can analyze an LTI system's frequency response by analyzing the DFT of its impulse response.
#

H = np.fft.fft(np.ones(5)/5)
plt.stem(np.fft.fftshift(np.fft.fftfreq(5)), np.fft.fftshift(np.abs(H)))
plt.title('DFT of Mean Filter (Length=5)')
plt.ylabel('Magnitude')
plt.xlabel('Frequency (cycles/sample)')
plt.savefig('frequency-response.png')
plt.close()

# lit unskip
# lit execute
# lit text
