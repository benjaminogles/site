
# lit text
#
# Mathematical Background
# -----------------------
#
# Complex numbers have the form `x + jy` where the number `x` is called the real part, the number `y` is called the imaginary part and `j` has the strange property that `j * j = -1`.
# Plotting the real and imaginary parts as a point on a 2D plane gives a graphical view of how the number's magnitude and phase is defined.
# The length of the line connecting the origin and the point is the number's magnitude.
# The angle between the positive real axis and that line is the number's phase.
# The sign of the number's phase indicates whether it is measured in a clockwise (positive) or counter-clockwise (negative) direction.
#
# lit skip

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.plot([0, 1], [0, 1])
plt.text(.5, .6, '$R$')
plt.plot(1, 1, '.', markersize=10)
plt.text(1, 1.1, '$(x,y)$')
plt.plot([0, 1], [0, 0])
plt.text(.5, -.1, '$x = rcos(\\theta)$')
plt.plot([1, 1], [0, 1])
plt.text(1.05, .5, '$y = rsin(\\theta)$')
plt.gca().add_patch(patches.Arc((0, 0), .8, .8, theta2=45))
plt.text(.4 * np.cos(np.pi/8), .4 * np.sin(np.pi/8), '$\\theta$')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.gca().spines['top'].set_visible(0)
plt.gca().spines['right'].set_visible(0)
plt.gca().spines['bottom'].set_position('zero')
plt.gca().spines['left'].set_position('zero')
plt.xticks([])
plt.yticks([])
plt.savefig('complex-number.png')
plt.close()

# lit execute
# lit unskip
# lit text
#
# As indicated in the figure, `x` and `y` can be rewritten in terms of the trigonometric functions `sin` and `cos`.
# This gives us a way to write `x + jy` in terms of its magnitude and phase explicitly: `R(cos(θ) + jsin(θ))`.
# We can also keep the explicit expression of magnitude and phase without using trigonometric functions by rewriting according to Euler's formula: `cos(θ) + jsin(θ) = exp(jθ)`.
# Any complex number with magnitude `R` and phase `θ` can be written as `Rexp(jθ)`.
#
# Many common operations on complex numbers are simple to perform and understand in this exponential form.
# Multiplying two complex numbers will multiply their magnitudes and add their phases e.g. `Aexp(jθ) * Bexp(jϕ) = ABexp(j(θ + ϕ))`.
# Complex conjugation simply negates the phase of a number e.g `conj(Aexp(jθ)) = Aexp(-jθ)`.
# So multiplying a complex number by its own conjugate always results in a scalar value equal to its magnitude squared since the phase goes to `0` and reduces the exponential to `1`.
