
# lit skip
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches

# lit unskip
# lit text
#
# Complex Samples
# ===============
#
# In this post I describe how complex numbers can be used to represent samples of a real-valued sinusoidal signal.
#
# Basics
# ------
#
# Complex numbers have the form `x + jy` where `x` is called the real part, `y` is called the imaginary part and `j*j = -1`.
# If we plot the point `(x, y)` on a 2D plane, it is clear that `x + jy` can also be defined in terms of its magnitude `A` (measured from the origin) and phase angle `θ` (measured from the positive horizontal axis).
#

# lit skip
x, y = 1, 1
plt.scatter([x], [y])
plt.text(x+.05, y+.05, '$(x, y)$')
plt.plot([0, 1], [0, 1])
plt.text(0.5, 0.6, '$A$')
plt.gca().add_patch(plt_patches.Arc([0,0], 0.5, 0.5, angle=0, theta1=0, theta2=45))
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

# Seed random number generator
import numpy as np
rng = np.random.default_rng(seed=1234)
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
# A bit more trigonometry takes us the other way: computing `x` and `y` from `A` and `θ`.
#

assert np.isclose(np.real(x_p_jy), A * np.cos(θ))
assert np.isclose(np.imag(x_p_jy), A * np.sin(θ))

# lit execute
# lit text
#
# At this point, it is clear that complex numbers are made from the same stuff as real-valued sinusoidal signals.
# The real-valued signal is defined by a time-varying amplitude and phase applied to the cosine or sine functions.
# The complex-valued equivalent is defined in two parts (real and imaginary) by the same time-varying amplitude and phase applied to the cosine and sine functions respectively.
#
# Euler's Formula
# ---------------
#
# It turns out that we can also construct `x + jy` in terms of `A` and `θ` by using the exponential function.
#

assert np.isclose(x_p_jy, A * np.exp(1j * θ))

# lit execute
# lit text
#
# When given an imaginary number, the exponential function simply returns a complex number with that phase in radians.
# We already saw that we can obtain this complex number by way of the cosine and sine functions so it follows that the exponential, cosine and sine functions would share a close relationship.
#

# Generate real and complex samples from A and θ
x = A * np.cos(θ)
y = A * np.sin(θ)
z = A * np.exp(1j * θ)
# Start with Euler's formula
assert np.isclose(z, x + 1j * y)
# Complex conjugate of both sides
assert np.isclose(np.conj(z), x - 1j * y)
# Add this equation to the one above it
assert np.isclose(z + np.conj(z), 2 * x)
# Solve for the cosine term
assert np.isclose((z + np.conj(z)) / 2,  x)
# Subtracting the two equations instead allows solving for sine
assert np.isclose((z - np.conj(z)) / 2j, y)

# lit execute
# lit text
#
# The sine and cosine functions are composed of a complex exponential combined with its own conjugate.
# Adding or subtracting a complex number to or from its conjugate reduces the imaginary part to `0`, leaving only the real-valued cosine or sine term.
# Otherwise, the conjugate term is completely redundant (just a sign difference in the imaginary part) so we can safely discard it and only analyze the first complex term.
#
# Why?
# ----
#
# Hopefully it is clear that complex numbers are a completely valid format for the samples of a real-valued sinusoidal signal.
# But why would we prefer this format over real-valued samples in certain applications?
#
# The magnitude and phase of a complex sample can be considered an estimate of a continuous signal's magnitude and phase at that sampling instant.
# This information is of special interest when implementing communication systems that modulate the magnitude and phase of a pure sinusoid to carry user data.
# It is not nearly as straight foward to estimate the instantaneous magnitude and phase of a signal from its real-valued samples.
# Even estimating the constant magnitude and frequency of a pure sinusoid would require analyzing a block of real-valued samples by e.g. noting extreme values and counting zero-crossings.
#
# How?
# ----
#
# How do we sample a real-valued signal to obtain complex samples?
# There is more than one way to do this but it is often built into the frequency translation process of radio hardware.
# The basic idea of frequency translation is to multiply an input signal with a pure sinusoid generated by a local oscillator (LO).
#

# Arbitrary carrier frequency
fc = rng.uniform(3e6, 30e6)
# Arbitrary LO frequency
lo = rng.uniform(3e6, 30e6)
# Arbitrary point in time
t = rng.uniform(0, 1e-3)
# Input signal value (we are interested in A and θ)
x = A * np.cos(2 * np.pi * fc * t + θ)
# LO signal value
l = np.cos(2 * np.pi * lo * t)
# Difference frequency component
d = np.cos(2 * np.pi * (fc-lo) * t + θ)
# Sum frequency component
s = np.cos(2 * np.pi * (fc+lo) * t + θ)
# Product to sum identity
assert np.isclose(x * l, A/2 * (d + s))

# lit execute
# lit text
#
# The multiplication results in copies of the input signal at two shifted frequencies.
# When tuning the radio, the local oscillator frequency is chosen so that one of the shifted frequencies always matches a known and fixed intermediate frequency.
# Fixing an intermediate frequency makes it easier to design the rest of the hardware which will filter, amplify and then sample the shifted input signal.
#
# The example above used a real-valued local oscillator and would produce real-valued samples.
# If we use two local oscillators, we can effect frequency translation by complex multiplication.
#

# Local oscillator for real part
r = np.cos(2 * np.pi * lo * t)
# Sum and difference images of
rdiff = np.cos(2 * np.pi * (fc-lo) * t + θ)
rsum = np.cos(2 * np.pi * (fc+lo) * t + θ)
assert np.isclose(x * r, A/2 * (rdiff + rsum))

# Local oscillator for imaginary part
i = np.sin(2 * np.pi * lo * t)
# Sum and difference images
idiff = - np.sin(2 * np.pi * (fc-lo) * t + θ)
isum = np.sin(2 * np.pi * (fc+lo) * t + θ)
assert np.isclose(x * i, A/2 * (idiff + isum))

# Conceptually, we get complex sum and difference components
d = rdiff + 1j * idiff
s = rsum + 1j * isum
assert np.isclose(x * (r + 1j * i), A/2 * (d + s))

# lit execute
# lit text
#
# Note that this process does not require special hardware, it just requires two branches of hardware to handle the real and imaginary parts.
# The local oscillators use the same frequency but are offset in phase by `90` degrees.
# After the signals in the real and imaginary branches of hardware are filtered and amplified, their samples are interleaved as the real and imaginary parts of the signal's complex samples.
#

