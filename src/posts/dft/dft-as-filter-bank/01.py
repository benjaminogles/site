
# lit skip
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches

# lit unskip
# lit text
#
# Understanding the Discrete Fourier Transform (DFT)
# ==================================================
#
# In this post, I use a simple example problem to motivate the derivation of the DFT as a bank of bandpass filters.
#
# Example Problem
# ---------------
#
# We will generate several arrays of complex samples, each one representing a pure sinusoid oscillating at a unique, constant frequency.
# We will then combine these arrays by adding aligned samples together.
# Our task will be to estimate the individual frequency of each wave by analyzing the combined signal.
#

# Seed random number generator
import numpy as np
rng = np.random.default_rng(seed=184027)
# Pick number of waves to sample
nwaves = 4
# Pick number of samples to generate
nsamples = 128
# Sample indices
t = np.arange(nsamples)
# Bound cycle length in samples for smoother plots
min_wavelength_samps = 10
# Pick random frequency for each wave (cycles/sample)
min_freq = -1.0 / min_wavelength_samps
max_freq = 1.0 / min_wavelength_samps
freqs = rng.uniform(min_freq, max_freq, nwaves)
# Generate samples for each wave
waves = [np.exp(2j * np.pi * f * t) for f in freqs]
# Combine
signal = np.sum(waves, axis=0)

# lit skip
plt.plot(np.real(signal), label='real')
plt.plot(np.imag(signal), label='imag')
plt.legend()
plt.title('signal')
plt.xlabel('sample')
plt.savefig('signal.png')
plt.close()

# lit unskip
# lit text
#
# This is what the real and imaginary parts of our signal look like.
#
# lit execute
#
# If you are used to frequency expressed in Hertz, note that I've simply factored out physical units by setting our sample spacing equal to 1 so that sample index can be used as a proxy for e.g. time.
# In an actual application, you could scale our normalized frequency (cycles/sample) by the sampling rate (samples/second) to get back to Hertz (cycles/second).
#
# I have biased the generation of the example problem towards lower frequencies (in an absolute value sense) because more samples per cycle makes some of the plots look nicer.
# Per Nyquist, the actual bounds in frequency that our complex samples could accurately represent is `(-0.5, 0.5)`.
# As an example, if you tried to construct a wave with a frequency of `1.0` cycles/sample, then it would be indistinguishable from a `0` cycles/sample wave because the phase of each sample would be an integer multiple of `2π` which yields the same result as when the phase of each sample is `0`.
# Other exceedingly low or high frequencies will similarly yield the same set of samples as a wave within our `(-0.5, 0.5)` frequency range (possibly on the other side of `0`).
#
# Mathematically, frequency is simply defined as the derivative of phase (one cycle is `2π` radians) so nothing is stopping it from being negative.
# On a more practical level, I will show later on how you can shift the phases of complex samples to center an arbitrary frequency at the `0` cycles/sample level.
# Then the negative frequencies simply represent frequencies lower than the centered frequency.
#
# The Anatomy of a (Complex) Wave
# -------------------------------
#
# How can complex samples represent a real-valued sinusoid?
#
# As a quick review, complex numbers take the form `x + jy` where `x` is called the real part, `y` is called the imaginary part and `j*j = -1`.
# If we plot the point `(x, y)` on a 2D plane, it is clear that `x + jy` can be alternately defined in terms of a magnitude `A` (measured from the origin) and a phase angle `θ` (measured from the positive horizontal axis).
#

# lit skip
x, y = 1, 1
plt.scatter([x], [y])
plt.text(x+.05, y+.05, '$(x, y)$')
plt.plot([0, 1], [0, 1])
plt.text(0.5, 0.6, '$A$')
plt.gca().add_patch(plt_patches.Arc([0,0], 0.5, 0.5, 0, 0, 45))
plt.text(0.25, 0.1, '$\\theta$')
plt.xlim(-1.25, 1.25)
plt.ylim(-1.25, 1.25)
plt.xticks([])
plt.yticks([])
plt.gca().spines['left'].set_position('center')
plt.gca().spines['top'].set_position('center')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_color('none')
plt.title('a complex number')
plt.savefig('complex-number.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# The values of `A` and `θ` are also easily computed from `x` and `y` using basic trigonometry.
#

# Generate x + jy with random x and y
x_p_jy = rng.uniform() + 1j * rng.uniform()
# Compute A and θ from their definitions
A = np.sqrt(np.real(x_p_jy)**2 + np.imag(x_p_jy)**2)
θ = np.arctan2(np.imag(x_p_jy), np.real(x_p_jy))
# Verify against library functions
assert np.isclose(A, np.abs(x_p_jy))
assert np.isclose(θ, np.angle(x_p_jy))

# lit execute
# lit text
#
# We can construct `x + jy` in terms of `A` and `θ` explicitly using the exponential function.
#

assert np.isclose(x_p_jy, A * np.exp(1j * θ))

# lit execute
# lit text
#
# Raising the number `e` to a power with an imaginary exponent is a bit weird but the result is straightforward: a unit magnitude complex number with the phase provided in the exponent (which we then scale by `A`).
# This is how we generated the samples for our example problem.
# A constant frequency in cycles per sample was converted to radians per sample and multiplied by each sample index to arrive at phase in radians.
#
# We could have generated real-valued samples by passing these phases to the cosine or sine functions instead of the exponential function.
# But those samples would not have directly encoded the phases and magnitude of the wave at each sampling instant the way complex samples do.
#
# Complex exponentials are more than just a convenient proxy for their real-valued counterparts though.
# The following program further details the simple relationship between the two formats.
#

# Random phases and magnitudes
phases = rng.uniform(0, 2 * np.pi, nsamples)
magnitudes = rng.uniform(-1, 1, nsamples)
# Generate a cosine, sine and complex wave
a = magnitudes * np.cos(phases)
b = magnitudes * np.sin(phases)
c = magnitudes * np.exp(1j * phases)
# Start with Euler's formula
assert np.allclose(c, a + 1j * b)
# Complex conjugate of both sides
assert np.allclose(c.conj(), a - 1j * b)
# Add this equation to the one above it
assert np.allclose(c + c.conj(), 2 * a)
# Solve for the cosine wave expression
assert np.allclose((c + c.conj()) / 2,  a)
# Subtracting the two equations instead allows solving for sine
assert np.allclose((c - c.conj()) / 2j, b)
# If we got here, every test passed
print('Assertions passed')

# lit text
#
# We might as well run this to check that I wrote everything down correctly.
#
# lit execute
#
# So we can see that sine and cosine waves are composed of a complex exponential combined with its own conjugate.
# This combination cancels the imaginary part of the complex wave, leaving the real sine or cosine.
# The information in the complex conjugate term is completely redundant (just a difference in sign on the imaginary part) so we can safely discard it.
# The remaining complex exponential gives us a direct view into the magnitudes of each sample, the phases of each sample and even the cosine (real part) and sine (imaginary part) of those magnitudes and phases.
#
# Back to the Example Problem
# ---------------------------
#
# Before we start attempting to solve the problem, I want to give some more context as to why it is a useful example.
# We'll see that the structure of the problem is obviously useful for introducing the DFT.
# But it is not contrived solely for this purpose.
# It is also closely related to practical application.
#
# Pure sinusoids can be modulated and transmitted with radio hardware to implement wireless communication systems.
# Multiple transmitters can be active at the same time if they are transmitting at different frequencies with sufficient separation.
# Our example problem simulates the physical phenomenon (called wave superposition) that occurs when waves from multiple transmitters meet in the air.
# The waves combine such that the displacement of the combined wave at every point is equal to the sum of the individual wave displacements.
#
# If you used another set of radio hardware to receive and digitize this combined waveform, you would get something that looks like our example problem.
# You could use the DFT to see which frequencies are active.
# As a sneak peek, here is a zoomed in look at the DFT of our combined signal.
#
# lit skip

S = np.fft.fftshift(np.fft.fft(signal, nsamples*2))/(nsamples*2)
f = np.fft.fftshift(np.fft.fftfreq(nsamples*2))
plt.plot(f[nsamples//2:-nsamples//2], np.real(S * S.conj())[nsamples//2:-nsamples//2])
plt.title('DFT(signal)')
plt.xlabel('frequency (cycles/sample)')
plt.ylabel('mag^2')
plt.savefig('dft.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# Step 1
# ------
#
# With that background out of the way, we can get started on solving this example problem.
# As a first step, we could pick a frequency and try to detect whether a wave at that frequency was summed into our signal.
# The simplest frequency to reason about is `0` cycles/sample because its wave is just a sequence of one repeated sample:

phi = rng.uniform() # random phase offset
zero_freq_wave = magnitudes * np.exp(1j * (2 * np.pi * 0 * t + phi))
repeated_samples = magnitudes * np.exp(1j * phi)
assert np.allclose(zero_freq_wave, repeated_samples)

# lit text
#
# When summed into another signal, this wave would simply shift each sample up or down by a constant.
# This means that we can detect a wave at `0` cycles/sample in our signal by computing the signal's mean value.
#

# lit skip
assert np.abs(np.mean(signal)) < 0.2, "did the rng seed change?"
# lit unskip

print(np.mean(signal))

# lit text
#
# In this case, the mean value our signal is near `0` in both the real and imaginary parts.
# We would expect a larger mean value if one of our summed waves had a frequency of `0`.
#
# lit execute
#
# But wouldn't we also expect a mean value of _exactly_ `0` if none of our summed waves had a frequency of `0`.
# In this case, all of our waves should be oscillating evenly around `0`.
#
