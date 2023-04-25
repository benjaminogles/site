
# lit text
#
# Linear and Time-Invariant (LTI) Systems
# =======================================
#
# **Published: 2023/04/25**
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
plt.stem(idx, x, basefmt=' ', use_line_collection=True)
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
plt.stem(idx, np.convolve(x, h, mode='same'), basefmt=' ', use_line_collection=True)
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
axs[0][0].stem(idx, x, basefmt=' ', use_line_collection=True)
axs[0][1].stem(idx, np.convolve(x, h, mode='same'), basefmt=' ', use_line_collection=True)
axs[1][0].stem(idx, -1 * np.roll(x, -1), basefmt=' ', use_line_collection=True)
axs[1][1].stem(idx, np.convolve(-1 * np.roll(x, -1), h, mode='same'), basefmt=' ', use_line_collection=True)
axs[2][0].stem(idx, 2 * np.roll(x, 1), basefmt=' ', use_line_collection=True)
axs[2][1].stem(idx, np.convolve(2 * np.roll(x, 1), h, mode='same'), basefmt=' ', use_line_collection=True)
axs[3][0].stem(idx, x + -1 * np.roll(x, -1) + 2 * np.roll(x, 1), basefmt=' ', use_line_collection=True)
axs[3][1].stem(idx, np.convolve(x + -1 * np.roll(x, -1) + 2 * np.roll(x, 1), h, mode='same'), basefmt=' ', use_line_collection=True)
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
# Computing a weighted sum with a sliding set of weights is called convolution.
#
# In general, the length of an LTI system impulse response may be infinite (IIR), which makes digital implementation more complicated than basic convolution.
# But many digital LTI systems can be designed to have a finite impulse response (FIR) so that they can be implemented directly as convolution with that response.
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
# In another [post](/posts/dft/), I showed that an input signal `x` convolved with an impulse response `h` (first column in a circulant matrix) has a DFT equivalent to the element-wise multiplication of `dft(x)` and `dft(h)`.
# So we can analyze an LTI system's frequency response by analyzing the DFT of its impulse response.
#

hlen = 5
H = np.fft.fft(np.ones(hlen)/hlen)

# lit skip

plt.stem(np.fft.fftshift(np.fft.fftfreq(hlen)), np.fft.fftshift(np.abs(H)), basefmt=' ', use_line_collection=True)
plt.title('Mean Filter Frequency Response (Length=5)')
plt.ylabel('Magnitude')
plt.xlabel('Frequency (cycles/sample)')
plt.savefig('frequency-response.png')
plt.close()

# lit unskip
# lit text
#
# The mean filter's frequency response singles out a single entry in the DFT, corresponding to a frequency of zero cycles per sample.
#
# lit execute
#
# Note that keeping one non-zero entry in the input DFT is not the same as perfectly filtering for one frequency.
# As discussed in the DFT post, power at any frequency that does not divide into an integer number of cycles over the length of the DFT will contribute some power to each entry of the result.
# We can see this by zero-padding the impulse response and analyzing it with a longer DFT.
# This does not change the behavior of the system at all but gives us better resolution in the frequency response.
#

H = np.fft.fft(np.ones(hlen)/hlen, n=hlen*10)

# lit skip
#
plt.stem(np.fft.fftshift(np.fft.fftfreq(hlen*10)), np.fft.fftshift(np.abs(H)), basefmt=' ', use_line_collection=True)
plt.title('Mean Filter Frequency Response (Length=5)')
plt.ylabel('Magnitude')
plt.xlabel('Frequency (cycles/sample)')
plt.savefig('frequency-response-fine.png')
plt.close()

# lit unskip
# lit text
#
# As expected, we see a peak at zero cycles per sample and minima at other frequencies that divide into an integer number of cycles over the length of the DFT.
#
# lit execute
