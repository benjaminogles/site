
# lit text
#
# Nyquist Frequency
# =================
#
# In this post, I describe the limits of representing a sinusoidal signal by its discrete samples.
#
# Definition
# ----------
#
# The Nyquist frequency, named after Harry Nyquist, is the highest frequency that a sampler can sample at a given sampling rate without introducing distortion due to aliasing in its output.
#
# Example
# -------
# 
# The following program shows that a sequence of samples will always represent an infinite family of sinusoidal signals.
#

import numpy as np

# Arbitrary sampling rate
fs = 1
# Sample spacing
T = 1 / fs
# Arbitrary number of samples
nsamples = 10
# Sample index multiplied by sample spacing
nT = np.arange(nsamples) * T
# Simulate continuous time with much higher rate
t = np.arange(30*nsamples) * T/30
# Arbitrary frequency
f1 = 0.2 * fs
# Simulate continuous signal at f1
a = np.cos(2 * np.pi * f1 * t)
# Real valued samples of a at sampled at fs
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
# The plot shows that sampling `b` at `fs` will yield the same sequence of values as sampling `a` when their frequencies are separated by any integer multiple of `fs`.
#

b_n = np.cos(2 * np.pi * f2 * nT)
assert np.allclose(a_n, b_n)

# lit execute
# lit text
#
# The really boils down to the trigonometric identity `cos(x) == cos(x + 2πn)` for any value of `n`.
#

