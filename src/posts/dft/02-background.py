
# lit text
#
# Mathematical Background
# -----------------------
#
# Complex numbers have the form `x + jy` where the number `x` is called the real part, the number `y` is called the imaginary part and `j` has the strange property that `j * j = -1`.
# Plotting complex numbers as points on a 2D plane gives a graphical view of how their magnitude and phase are defined.
# The length of the line between the origin and the point is the number's magnitude.
# The angle between the positive real axis and the line is the number's phase.
#
# lit skip

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.plot([0, 1], [0, 1])
plt.text(.5, .6, '$r$')
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
