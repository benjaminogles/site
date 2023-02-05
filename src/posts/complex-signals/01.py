
# lit skip
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import numpy as np

# lit unskip
# lit text
#
# Complex Signals
# ===============
#
# **Published: 2022/12/15**
#
# In this post, I briefly define complex numbers and describe how they are used in many signal processing applications.
#
# Basics
# ------
#
# Complex numbers have the form `x + jy` where `x` and `y` are real numbers and `j` is the imaginary unit: `j*j = -1`.
# We call `x` the real part and `y` the imaginary part of the number.
#

x = 1
y = 1
x_p_jy = x + 1j * y

# lit text
#
# If we plot the point `(x, y)` on a 2D plane it is clear that complex numbers are equally well defined in terms of a magnitude, measured from the origin, and a phase angle, measured from the positive horizontal axis.
#

# lit skip
θ = np.arctan2(y, x)
diameter = min(abs(x), abs(y))/2
plt.scatter([x], [y])
plt.text(x+.05, y+.05, '$(x, y)$')
plt.plot([0, x], [0, y])
plt.text(x/3.5, y/2+0.1, '$\\sqrt{x^2 + y^2}$')
plt.gca().add_patch(plt_patches.Arc([0,0], diameter, diameter, angle=0, theta1=0, theta2=np.degrees(θ)))
plt.text(diameter/2 * np.cos(θ/2) + 0.02, diameter/2 * np.sin(θ/2), '$tan^{-1}(\\frac{y}{x})$')
plt.plot([0, x], [0, 0])
plt.text(x/2, -0.15, '$x$')
plt.plot([x, x], [0, y])
plt.text(x+0.05, y/2-0.05, '$y$')
plt.xlim(-abs(x)-0.25, abs(x)+0.25)
plt.ylim(-abs(y)-0.25, abs(y)+0.25)
plt.xticks([])
plt.yticks([])
plt.gca().spines['left'].set_position('center')
plt.gca().spines['top'].set_position('center')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_color('none')
plt.title('a complex number')
plt.savefig('complex-number-1.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# As indicated in the plot, the magnitude and phase of a complex number can be computed from its real and imaginary parts using just a bit of basic trigonometry.
#

# Compute A and θ from their definitions
import numpy as np
A = np.sqrt(x**2 + y**2)
θ = np.arctan2(y, x)
# Verify against library functions
assert np.isclose(A, np.abs(x_p_jy))
assert np.isclose(θ, np.angle(x_p_jy))

# lit execute
# lit text
#
# We can also go the other way and compute `x` and `y` from `A` and `θ`.
# These are completely interchangeable representations of the same complex number.
# Here is the plot of `(x, y)` again but labeled to show the definitions of `x` and `y` in terms of `A` and `θ`.
#

# lit skip
θ = np.arctan2(y, x)
diameter = min(abs(x), abs(y))/2
plt.scatter([x], [y])
plt.text(x+.05, y+.05, '$(x, y)$')
plt.plot([0, x], [0, y])
plt.text(x/2.5, y/2+0.1, '$A$')
plt.gca().add_patch(plt_patches.Arc([0,0], diameter, diameter, angle=0, theta1=0, theta2=np.degrees(θ)))
plt.text(diameter/2 * np.cos(θ/2) + 0.02, diameter/2 * np.sin(θ/2), '$\\theta$')
plt.plot([0, x], [0, 0])
plt.text(x/2, -0.15, '$Acos(\\theta)$')
plt.plot([x, x], [0, y])
plt.text(x+0.05, y/2-0.05, '$Asin(\\theta)$')
plt.xlim(-abs(x)-0.25, abs(x)+0.25)
plt.ylim(-abs(y)-0.25, abs(y)+0.25)
plt.xticks([])
plt.yticks([])
plt.gca().spines['left'].set_position('center')
plt.gca().spines['top'].set_position('center')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_color('none')
plt.title('a complex number')
plt.savefig('complex-number-2.png')
plt.close()

# lit unskip
# lit execute

assert np.isclose(x, A * np.cos(θ))
assert np.isclose(y, A * np.sin(θ))

# lit execute
# lit text
#
# Relationship to Signals
# -----------------------
#
# In many signal processing applications, we need to analyze signals of the form `x(t) = A(t) * cos(θ(t))` where `A(t)` and `θ(t)` are arbitrary real-valued functions of `t` (e.g. time).
# It is immediately apparent that we can represent this signal with complex numbers.
#

rng = np.random.default_rng()
# Generate random A(t)
A = rng.uniform(0, 1, size=10)
# Generate random θ(t)
θ = rng.uniform(-np.pi, np.pi, size=10)
# A(t) * cos(θ(t))
x = A * np.cos(θ)
# A(t) * sin(θ(t))
y = A * np.sin(θ)
# Complex samples
x_p_jy = x + 1j * y
# We can recover A(t)
assert np.allclose(np.abs(x_p_jy), A), str(list(A)) + " != " + str(list(np.abs(x_p_jy)))
# We can recover θ(t)
assert np.allclose(np.angle(x_p_jy), θ)
# We can recover x(t)
assert np.allclose(np.real(x_p_jy), x)
# The complex signal has all we need

# lit execute
# lit text
#
# How?
# ----
#
# The program above showed that we can represent the signal `x(t) = A(t) * cos(θ(t))` with complex numbers if we can compute `y(t) = A(t) * sin(θ(t))` to form `x(t) + j * y(t)`.
# This is easy enough if we know `A(t)` and `θ(t)` separately.
# It is also possible to compute `y(t)` from `x(t)` alone as well.
# But I will cover that in a separate post.
#
# Why?
# ----
#
# In many applications, all of the important information resides in `A(t)` and `θ(t)` rather than `x(t)` itself.
# In these cases, we need a way to quickly and accurately recover `A(t)` and `θ(t)` from `x(t)`.
# As mentioned above, this is done by computing `y(t) = A(t) * sin(θ(t))` and analyzing the complex-valued signal `x(t) + j * y(t)` instead.
#
# Euler's Formula
# ---------------
#
# Rather than writing `x + jy`, it is often convenient to write complex numbers explicitly in terms of `A` and `θ` with the exponential function.
#

z = A * np.exp(1j * θ)
assert np.allclose(x_p_jy, z)

# lit execute
# lit text
#
# When given an imaginary number, the exponential function returns a complex number with that phase in radians.
# The following snippet further details the relationship between the expressions `x + jy` and `A * exp(jθ)` in this example.
#

# Start with Euler's formula
assert np.allclose(z, x + 1j * y)
# Complex conjugate of both sides
assert np.allclose(np.conj(z), x - 1j * y)
# Add this equation to the one above it
assert np.allclose(z + np.conj(z), 2 * x)
# Solve for the cosine term
assert np.allclose((z + np.conj(z)) / 2,  x)
# Solve for the sine term
assert np.allclose((z - np.conj(z)) / 2j,  y)

# lit execute
# lit text
#
# This shows that the signals `x(t) = A(t) * cos(θ(t))` and `y(t) = A(t) * sin(θ(t))` are composed of complex-valued signals in conjugate pairs.
# Converting a real-valued signal to a complex-valued signal is just a matter of removing the redundant conjugate term.
# When we add `x(t)` to `j * y(t)`, the redundant conjugate terms cancel out and we are left with `z(t) = A(t) * exp(j * θ(t))`.
#
# Generalizing
# ------------
#
# So far, we have only discussed signals of the simple form `A(t) * cos(θ(t))`.
# What if we are working with a sum of many such signals?
# Luckily, a sum of these signals takes on the exact same form.
#

# Random B(t)
B = rng.uniform(0, 1, size=10)
# Random ϕ(t)
ϕ = rng.uniform(-np.pi, np.pi, size=10)
# Real-valued samples
r = B * np.cos(ϕ)
# Complex-valued sample
w = B * np.exp(1j * ϕ)
# Real-valued samples of the summed signal
s = x + r
# Unpack definitions of x and r in terms of w and z
assert np.allclose(s, (z + np.conj(z) + w + np.conj(w)) / 2)
# Rearrange; s is composed of complex numbers and their conjugates
assert np.allclose(s, (z+w + np.conj(z+w)) / 2)
# Write s in the form A(t) * cos(θ(t))
assert np.allclose(s, np.abs(z+w) * np.cos(np.angle(z+w)))

# lit execute
# lit text
#
# So everything in this post also applies to arbitrary sums of signals in this form.
#
