
# lit skip
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches

# lit unskip
# lit text
#
# Complex Signals
# ===============
#
# **Publish Date: 2022/12/12**
#
# In this post, I describe how complex numbers can be used to represent real-valued signals of the form `A(t) * cos(θ(t))` where `A(t)` and `θ(t)` are arbitrary real-valued functions of `t` (e.g. time).
# The choice of cosine over sine is completely arbitrary and makes no difference in the details of this post since the two functions are simply shifted copies of each other (we can always write sine in terms of cosine).
# The interesting information resides in the functions `A(t)` and `θ(t)` which respectively provide the instantaneous magnitude and phase of the signal.
#
# Given the magnitude and phase of a signal at a particular instant, we can always compute the cosine or sine sample value.
# It is not nearly as straightforward a task to go the other way and compute a particular real-valued sample's magnitude and phase.
# In fact, one of the main advantages of complex numbers in this context is that they _do_ provide a very straightforward representation of instantaneous magnitude and phase.
#
# Throughout this post, we will consider a single random sample of a signal with the form `A(t) * cos(θ(t))` to understand how it can be represented by a complex number.
#

# Seed random number generator
import numpy as np
rng = np.random.default_rng(seed=1234)
# Generate a random magnitude and phase
A = rng.uniform(0.5, 2)
θ = rng.uniform(0, 2 * np.pi)

# lit text
#
# Basics
# ------
#
# Complex numbers have the form `x + jy` where `x` is called the real part, `y` is called the imaginary part and `j*j = -1`.
# They can be plotted on a 2D plane with `x` on the horizontal (real) axis and `y` on the vertical (imaginary) axis.
# If we draw a line from the origin to any point representing a complex number, it is clear that `x + jy` can also be defined in terms of a magnitude (measured from the origin) and a phase angle (measured from the positive real axis).
#
# This means that we can plot `A` and `θ` on a 2D plane to find the complex number representation of our sample with just a bit of trigonometry.
#

# lit skip
x = A * np.cos(θ)
y = A * np.sin(θ)
diameter = min(abs(x), abs(y))/2
plt.scatter([x], [y])
plt.text(x+.05, y+.05, '$(x, y)$')
plt.plot([0, x], [0, y])
plt.text(x/2, y/2+0.1, '$A$')
plt.gca().add_patch(plt_patches.Arc([0,0], diameter, diameter, angle=0, theta1=0, theta2=np.degrees(θ)))
plt.text(diameter/2 * np.cos(θ/2), diameter/2 * np.sin(θ/2) + 0.1, '$\\theta$')
plt.plot([0, x], [0, 0])
plt.text(x/1.5, -0.2, '$A * cos(\\theta)$')
plt.plot([x, x], [0, y])
plt.text(x-0.5, y/2-0.1, '$A * sin(\\theta)$')
plt.xlim(-abs(x)-0.25, abs(x)+0.25)
plt.ylim(-abs(y)-0.25, abs(y)+0.25)
plt.xticks([])
plt.yticks([])
plt.gca().spines['left'].set_position('center')
plt.gca().spines['top'].set_position('center')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_color('none')
plt.title('a complex sample')
plt.savefig('complex-sample.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# As shown in the plot, the real and imaginary parts of the complex number representation have simple definitions in terms of `A` and `θ`.
#

x = A * np.cos(θ)
y = A * np.sin(θ)
x_p_jy = x + 1j * y

# lit text
#
# And as mentioned above, the magnitude and phase of this sample are easily recovered from the complex number representation.
#

# Compute A and θ from their definitions
assert np.isclose(np.sqrt(x**2 + y**2), A)
assert np.isclose(np.arctan2(y, x), θ)
# Also verify against library functions
assert np.isclose(np.abs(x_p_jy), A)
assert np.isclose(np.angle(x_p_jy), θ)

# lit execute
# lit text
#
# Euler's Formula
# ---------------
#
# Rather than writing `x + jy`, it is often convenient to write complex numbers explicitly in terms of `A` and `θ` by way of the exponential function.
#

assert np.isclose(x_p_jy, A * np.exp(1j * θ))

# lit execute
# lit text
#
# When given an imaginary number, the exponential function returns a complex number with that phase in radians.
# The following snippet further details the relationship between the expressions `x + jy` and `A * exp(jθ)` in this example.
#

# Generate real and complex samples from A and θ
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
# This gives us another insight into why we can easily represent signals of the form `A(t) * cos(θ(t))` or `A(t) * sin(θ(t))` with complex numbers.
# Here, `x` and `y` have that form exactly and we can see that they are equal to the complex number `z` combined with its own complex conjugate.
# The conjugate term is completely redundant so we can simply drop it and reason about `z` alone.
#
# Generalizing
# ------------
#
# So far, we have discussed signals with the simple form `A(t) * cos(θ(t))`.
# We may also want to analyze the sum of many such signals.
# Luckily, the sum of any two signals in this form can also be expressed in the same way.
#

# Another random sample from some other signal
B = rng.uniform(0.5, 2)
ϕ = rng.uniform(0, 2 * np.pi)
# Real-valued representation
u = B * np.cos(ϕ)
# Complex-valued representation
w = B * np.exp(1j * ϕ)
# Real-valued sample of the summed signal
s = x + u
# Unpack definitions of x and u in terms of w and z
assert np.isclose(s, (z + np.conj(z) + w + np.conj(w)) / 2)
# Rearrange; s is composed of a complex number and its conjugate
assert np.isclose(s, (z+w + np.conj(z+w)) / 2)
# Write s in the form A(t) * cos(θ(t))
assert np.isclose(s, np.abs(z+w) * np.cos(np.angle(z+w)))

# lit execute
# lit text
#
# So everything in this post also applies to arbitrary sums of signals in this form.
#
# Converting from Real to Complex
# -------------------------------
#
# How do we convert a real-valued signal to complex if we don't know `A(t)` and `θ(t)`?
# There is actually more than one way to do this but it is often built into the frequency translation process of radio hardware.
# The basic idea of frequency translation in a radio is to multiply an input signal with a pure sinusoid generated by a local oscillator (LO).
#

# Arbitrary carrier frequency
fc = rng.uniform(3e6, 30e6)
# Arbitrary LO frequency
lo = rng.uniform(3e6, 30e6)
# Arbitrary point in time
t = rng.uniform(0, 1e-3)
# Input signal value (we are interested in A and θ)
v = A * np.cos(2 * np.pi * fc * t + θ)
# LO signal value
l = np.cos(2 * np.pi * lo * t)
# Difference frequency component
d = np.cos(2 * np.pi * (fc-lo) * t + θ)
# Sum frequency component
s = np.cos(2 * np.pi * (fc+lo) * t + θ)
# Product to sum identity
assert np.isclose(v * l, A/2 * (d + s))

# lit execute
# lit text
#
# The multiplication results in copies of the input signal at two shifted frequencies.
# When tuning the radio, the local oscillator frequency is chosen so that one of the shifted frequencies always matches a known and fixed intermediate frequency.
# Fixing an intermediate frequency makes it easier to design the rest of the hardware which will filter, amplify and then sample the shifted input signal.
#
# There may be multiple stages of frequency translation and the complex conversion can be done before or after sampling.
# The example above used one local oscillator and would produce real-valued samples.
# If we use two local oscillators, we can effect frequency translation by complex multiplication.
#

# Local oscillator for real part
r = np.cos(2 * np.pi * lo * t)
# Sum and difference images
rdiff = np.cos(2 * np.pi * (fc-lo) * t + θ)
rsum = np.cos(2 * np.pi * (fc+lo) * t + θ)
assert np.isclose(v * r, A/2 * (rdiff + rsum))

# Local oscillator for imaginary part
i = np.sin(2 * np.pi * lo * t)
# Sum and difference images
idiff = - np.sin(2 * np.pi * (fc-lo) * t + θ)
isum = np.sin(2 * np.pi * (fc+lo) * t + θ)
assert np.isclose(v * i, A/2 * (idiff + isum))

# Taken together, we get complex sum and difference components
d = rdiff + 1j * idiff
s = rsum + 1j * isum
# Conceptually, this is complex multiplication
assert np.isclose(v * (r + 1j * i), A/2 * (d + s))

# lit execute
# lit text
#
# Note that this process does not require special hardware, it just requires two branches of hardware to handle the real and imaginary parts.
# The local oscillators use the same frequency but are offset in phase by `90` degrees.
# After the signals in the real and imaginary branches of hardware are filtered and amplified, their samples are interleaved as the real and imaginary parts of the signal's complex samples.
# 
