
# lit text
#
# Nyquist Frequency
# =================
#
# **Published Date: 2022/12/12**
#
# In this post, I describe the limits of representing a continuous signal by its discrete samples.
# Such a representation is only useful if it is unique and reversible i.e. if we can interpolate the samples to get back to a good approximation of the original continuous signal.
# This becomes impossible when the continuous signal contains high frequency oscillations between any two samples.
# The following program shows an example of this problem using pure sinusoidal signals.
#

import numpy as np

# Arbitrary sampling rate
fs = 1
# Sample spacing
T = 1 / fs
# Arbitrary number of samples
nsamples = 10
# Sample indices
n = np.arange(nsamples)
# Sample indices multiplied by sample spacing
nT = n * T
# Simulate continuous time with much higher rate
t = np.arange(30*nsamples) * T/30
# Arbitrary frequency
f1 = 0.2 * fs
# Simulate continuous signal at f1
a = np.cos(2 * np.pi * f1 * t)
# Real valued samples of a sampled at fs
a_n = np.cos(2 * np.pi * f1 * nT)
# Any integer value of k would work
k = 1
# Frequency that aliases to f1 when sampled at fs
f2 = f1 + k * fs
# Simulate continuous signal at f2
b = np.cos(2 * np.pi * f2 * t)

# lit skip
import matplotlib.pyplot as plt
plt.plot(t, a, label='a')
plt.scatter(nT, a_n, label='a_n')
plt.plot(t, b, label='b')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.savefig('aliasing.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# The plot shows that sampling the signal `b` at rate `fs` will yield the same sequence of values as sampling signal `a` even though they are very different continuous signals.
#

b_n = np.cos(2 * np.pi * f2 * nT)
assert np.allclose(a_n, b_n)

# lit execute
# lit text
#
# This is because the higher frequency oscillations in `b` are completely missed by the lower rate sampling process.
# In general, the samples of a sinusoidal signal ambiguously represent an infinite family of sinusoidal signals separated by integer multiples of the sampling rate.
# This really boils down to the trigonometric identity `cos(x) == cos(x + 2πk)` for any value of `k`.
#

# Expand definition of f2
assert np.allclose(b_n, np.cos(2 * np.pi * (f1 + k * fs) * nT))
# Distribute
assert np.allclose(b_n, np.cos(2*np.pi*f1*nT + 2*np.pi*k*fs*nT))
# Simplify due to fs = 1/T
assert np.allclose(b_n, np.cos(2*np.pi*f1*nT + 2*np.pi*k*n))
# Phase shifts of 2πkn just wrap around the unit circle
# and do not affect the output of cos
assert np.allclose(b_n, np.cos(2*np.pi*f1*nT))

# lit execute
# lit text
#
# In order to avoid this problem, the sampling rate must exceed the difference between the lowest and highest frequency oscillations in the continuous signal.
# In other words, a sequence of samples can unambiguously represent continuous signals with frequency content limited to the range of `(-fs/2, fs/2)` shifted to any start frequency.
# This means our sampling rate is twice as large as the highest frequency we need to detect giving us more than a single sample on average in each peak and valley of those oscillations.
# The limiting frequencies `-fs/2` and `fs/2` are called Nyquist frequencies, named after the scientist Harry Nyquist.
#
