
# lit text
#
# Nyquist Frequency
# =================
#
# **Published: 2022/12/15**
#
# In this post, I describe the limits of representing a continuous signal by its discrete samples.
# Such a representation is only useful if it is unique and reversible i.e. if we can interpolate the samples to get back to a good approximation of the original continuous signal.
# This becomes impossible when the continuous signal contains high frequency oscillations between samples.
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
# This boils down to the trigonometric identity `cos(x) == cos(x + 2πk)` for any value of `k`.
#

# Expand definition of f2
assert np.allclose(b_n, np.cos(2 * np.pi * (f1 + k * fs) * nT))
# Distribute
assert np.allclose(b_n, np.cos(2*np.pi*f1*nT + 2*np.pi*k*fs*nT))
# Simplify due to fs = 1/T
assert np.allclose(b_n, np.cos(2*np.pi*f1*nT + 2*np.pi*k*n))
# Phase offsets of 2πkn just wrap around the unit circle
assert np.allclose(b_n, np.cos(2*np.pi*f1*nT))

# lit execute
# lit text
#
# This means that the samples of the signal `a` ambiguously represent an infinite family of other continuous sinusoidal signals with frequencies equal to `f1 + k * fs` for any integer `k`.
# The only way to work around this ambiguity is to limit the bandwidth of the continuous signal (the range of frequencies it contains) before sampling so that no two frequencies represented in the continuous signal are separated by an integer multiple of `fs`.
# Typically, we limit the bandwidth of the continuous signal to `(-fs/2, fs/2)` before sampling.
# The limiting frequencies `-fs/2` and `fs/2` are called Nyquist frequencies, named after the scientist Harry Nyquist who helped formalize a lot of this theory.
# This brings us full circle to the intuitive idea that our sampling process must adequately capture high frequency oscillations i.e. we need more than two samples, on average, in each cycle of any frequency contained in the continuous signal.
#
