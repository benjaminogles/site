
# lit text
#
# Discrete Fourier Transform
# ==========================
#
# In this post, I introduce the Discrete Fourier Transform (DFT) from a few different perspectives.
#
# Basic Matrix Perspective
# ------------------------
#
# Mathematically, a sequence of complex samples, such as the arbitrary and random one below, can be considered a _vector_.
# This vector belongs to a _vector space_ that contains all such length-`N` vectors.
#

# lit skip

import numpy as np

rng = np.random.default_rng(seed=1234)
def generate_complex_samples(N):
    """
    Return `N` samples of a complex gaussian random variable
    as an example input signal.
    """
    mean = 0
    scale = np.sqrt(0.5)
    real = rng.normal(mean, scale, N)
    imag = rng.normal(mean, scale, N)
    return real + 1j * imag

# lit unskip

N = 1024
x = generate_complex_samples(N)

# lit skip

import matplotlib.pyplot as plt
plt.plot(np.real(x), label='real')
plt.plot(np.imag(x), label='imag')
plt.legend(loc='best')
plt.xlabel('sample')
plt.ylabel('amplitude')
plt.title('x')
plt.savefig('x.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# The claim of the DFT is that we can construct any vector `x` in this space as a unique weighted sum (called a linear combination) of `N` pure sinusoids.
# A pure sinusoid, in vector form, is a sequence of complex samples that advance in phase according to the sinusoid's constant frequency.
#

def complex_sinusoid(freq, N):
    """
    Return `N` complex samples of a sinusoid oscillating at `freq`
    cycles/sample with an initial phase of `0`. Normalize the result
    to a unit length vector.
    """
    return np.exp(2j * np.pi * freq * np.arange(N)) / N

# lit text
#
# Note: see this [post](/posts/complex-signals/) for an overview of using complex numbers to represent samples of sinusoidal signals.
#
# A set of `N` complex sinusoid vectors with length `N` may form the columns of an `NxN` matrix.
#

def complex_sinusoids(N, freqs):
    """
    Return an `NxM` matrix where `M = len(freqs)`. Each column
    vector contains `N` samples of a sinusoid oscillating at the
    associated frequency in `freqs` (cycles/sample).
    """
    return np.column_stack([complex_sinusoid(f) for f in freqs])

# lit text
#
# To (hopefully) aid understanding, I'm going to postpone defining the `N` frequencies used to generate the columns of the DFT matrix and explaining why there must be `N` of them until we've had a chance to look at how the matrix is used in the DFT and inverse DFT formulas.
# 

def dft_freqs(N):
    pass # TODO

# lit text
#
# Assume that we have defined `dft_freqs(N)` and use it to generate the DFT matrix `A`.
# Also assume that we have a routine, `dft(x)` that computes the weights for generating `x` as a linear combination of the columns of `A`.
# Then the inverse DFT (computing `x` from `dft(x)`), is trivially derived as computing that linear combination.
#

def idft(X):
    """
    Inverse DFT: given `X = dft(x)`, return `x`.
    """
    N = len(X)
    A = complex_sinusoids(dft_freqs(N), N)
    return np.matmult(A, X)

# lit text
#
# If you're used to thinking of matrix multiplication results in an entry-by-entry way as row-column dot products, it is worth training your mind to also view the results in a column-by-column or row-by-row way as linear combinations of the matrix columns or rows depending on whether the matrix in question is on the left or right side of the expression.
# The matrix or vector on the other side of the expression contains the weights of the combination in its rows or columns depending on whether *it* appears on the left or right side of the expression.
# Eli Bendersky has a helpful visualization of these operations on his site [here](https://eli.thegreenplace.net/2015/visualizing-matrix-multiplication-as-a-linear-combination/).
#
# By studying `idft(X)`, we can start to answer why the DFT matrix must be an `NxN` matrix.
# To restate the claim of the DFT, we claim that `idft(X)` can generate every possible `x` in our `N`-dimensional vector space by combining the columns of `A` with a unique weight vector `X`.
#
# In linear algebra terms, we claim that the _column space_ of `A` is equivalent to our `N`-dimensional vector space, i.e. that the columns of `A` form a basis for the space.
# Although out of scope for this article, it is not too difficult to prove that every basis for an `N`-dimensional vector space has exactly `N` vectors.
# This [video](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/proof-any-subspace-basis-has-same-number-of-elements) from Kahn Academy can guide you most of the way by proving that every basis of a subspace must contain the same number of vectors (you can then use the columns of the appropriately sized identity matrix as an example basis for any subspace to complete the proof).
#
# We can use `idft(X)` to derive the expression for `dft(x)`.
# We have
#
# `x = idft(dft(x))`
#
# and expanding the definition of `idft(X)` gives
#
# `x = (A)dft(x)`.
#
# Multiplying by the inverse of `A` on both sides completes the derivation:
#
# `(A^-1)x = dft(x)`.
#
# Our previous claims about `A` imply that it is invertible.
# In fact, we can see that the invertibility of `A` is all we need to implement `dft(x)` and `idft(x)` correctly.
# Specifically, we can make the inverse of `A` equal to its conjugate transpose by making its columns _orthonormal_.
# Then `dft(x)` is easily implemented.
#

def dft(x):
    """
    Return the weights that generate `x` as a linear combination
    of the columns in the DFT matrix.
    """
    N = len(x)
    A = complex_sinusoids(dft_freqs(N), N)
    return A.conj().T @ x

# lit text
#
# Now we need to implement `dft_freqs(N)` to actually return `N` frequencies that generate orthonormal vectors.
# Two vectors are orthonormal if they are both unit vectors and they are _orthogonal_ i.e. their inner product is `0`.
# We already implemented `complex_sinusoids(N,freqs)` to always return unit vectors.
# Now we need to implement `dft_freqs(N)` to return `N` frequencies that generate vectors with.
#

def inner(a, b):
    "Return the inner product of vectors `a` and `b`"
    return np.dot(a.conj(), b)

# lit text
#
# Let's see how
#
# Two vectors are orthogonal if their _inner product_ is `0`.
#
