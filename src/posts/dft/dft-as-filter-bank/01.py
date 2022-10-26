
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
# Our task will be to estimate the individual frequency of each original wave by analyzing the combined signal.
#

# Seed random number generator
import numpy as np
rng = np.random.default_rng(seed=192678)
# Pick number of waves to sample
nwaves = 4
# Pick number of samples to generate
nsamples = 128
# Bound cycle length in samples for smoother plots
min_wavelength_samps = 10
# Pick random frequency for each wave (cycles per sample)
min_freq = -1.0 / min_wavelength_samps
max_freq = 1.0 / min_wavelength_samps
freqs = rng.uniform(min_freq, max_freq, nwaves)
# Generate samples for each wave
waves = [np.exp(2j*np.pi*f*np.arange(nsamples)) for f in freqs]
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
# But it is not solely contrived for this purpose.
# It has a strong connection to practical application.
#
# Pure sinusoids can be modulated and transmitted with radio hardware to implement wireless communication systems.
# Multiple transmitters can be active at the same time if they are transmitting at different frequencies with enough separation between them.
# Our example problem simulates the physical phenomenon (called wave superposition) that occurs when waves from multiple transmitters meet in the air.
# The waves combine such that the displacement of the combined wave at every point is equal to the sum of the individual wave displacements.
# 

