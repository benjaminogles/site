
# lit text
#
# Discrete Fourier Transform
# ==========================
#
# In this post, I derive the formula for the Discrete Fourier Transform (DFT) and examine it from a few different perspectives.
#
# Basic Matrix Derivation
# -----------------------
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

N = 128
x = generate_complex_samples(N)

# lit text
#
# The claim of the DFT is that we can construct this vector `x`, and any other such vector `x`, as a unique weighted sum (called a linear combination) of `N` pure sinusoids.
# The weights assigned to each sinusoid give us an indication of the frequency content in `x`.
# I wouldn't say this claim is obviously true, just by looking at our example `x`, so it is worth a thorough derivation.
#

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
# A pure sinusoid, in vector form, is a sequence of complex samples with phase angles that advance according to the sinusoid's constant frequency.
#

def complex_sinusoid(N, freq):
    """
    Return `N` complex samples of a sinusoid oscillating at `freq`
    cycles/sample with an initial phase of `0`. Normalize the result
    to a unit length vector.
    """
    return np.exp(2j * np.pi * freq * np.arange(N)) / np.sqrt(N)

# lit text
#
# Note: see this [post](/posts/complex-signals/) for an overview of using complex numbers to represent samples of sinusoidal signals.
#
# A set of `N` complex sinusoid vectors with length `N` may form the columns of an `NxN` matrix.
#

def complex_sinusoids(N, freqs):
    """
    Return an `NxM` matrix where `M = len(freqs)`. Each column
    vector contains `N` complex samples of a sinusoid oscillating
    at the associated frequency in `freqs` (cycles/sample).
    """
    return np.column_stack([complex_sinusoid(N, f) for f in freqs])

# lit text
#
# Assume, for now, that we have defined `dft_freqs(N)` so that we can use it to generate an `NxN` matrix `A` with columns containing complex samples of pure sinusoids.
# Also assume that we have a routine, `dft(x)` that computes the weights for generating `x` as a linear combination of the columns of `A`.
# Then the inverse DFT (computing `x` from `dft(x)`), is trivially derived as taking that linear combination <a id="footnote-1-ref" href="#footnote-1">[1]</a>.
#

def dft_freqs(N):
    """
    Return the `N` frequencies in cycles/sample used to generate
    the columns of the DFT matrix.
    """
    pass # TODO

def idft(X):
    """
    Inverse DFT: given `X = dft(x)`, return `x`.
    """
    N = len(X)
    A = complex_sinusoids(dft_freqs(N), N)
    return np.matmult(A, X)

# lit text
#
# We can now use `idft(X)` to derive the expression for `dft(x)`.
# We want
#
# `x = idft(dft(x))`
#
# and expanding the definition of `idft(X)` gives
#
# `x = (A)dft(x)`.
#
# Left multiplying by the inverse of `A` on both sides completes the derivation:
#
# `(A^-1)x = dft(x)`.
#
# Note that `(A^-1)` must be a left and right inverse of `A` (i.e. `A` must be invertible) because we need to left multiply the above equation by `A` to get back to the expression for `idft(x)`.
#
# So we have our answer on why `A` must be an `NxN` matrix: it must be a square matrix to be invertible and it must have `N` rows corresponding to the `N` samples in `x`.
# We could expound on the properties of `A` in linear algebra terms <a id="footnote-2-ref" href="#footnote-2">[2]</a> but it obviously isn't necessary for the derivation since we already have expressions for `dft(x)` and `idft(X)`.
# Perhaps the simplest way to think about deriving the DFT matrix is to lean on the algebra above and just think about choosing `dft_freqs(N)` so that `A` is invertible.
#
# If we choose `dft_freqs(N)` correctly, we can make `A` trivially invertible by making its columns _orthonormal_.
# Then, the inverse of `A` will be equal to its conjugate transpose and `dft(x)` is implemented easily.
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
# Two vectors are orthonormal if they are unit vectors and their inner product is `0`.
#

def inner(a, b):
    "Return the inner product of vectors `a` and `b`"
    return np.dot(a.conj(), b)

def is_unit(a):
    "Return whether `a` is a unit vector"
    return np.isclose(inner(a, a), 1.0)

def is_orthogonal(a, b):
    "Return whether `a` and `b` are orthogonal"
    return np.isclose(inner(a, b), 0.0)

# lit text
#
# We already implemented `complex_sinusoids(N,freqs)` to always return unit vectors.
# We need to determine which set of frequencies generate mutually orthogonal vectors.
#
# Recall (or see this [post](/posts/nyquist-frequency/)) that the samples of two sinusoids with the same initial phase will be exactly equal if the difference between their frequencies is an integer (when the units of frequency is cycles per sample).
# Two vectors cannot be orthogonal if they contain the same entries so we can immediately limit our search to a range of `1` cycles per sample.
# I'm not clever enough to derive the right set of frequencies from first principles so I'll just start with the first frequency in the range `[0, 1)` and do a brute force search for orthogonal frequencies.
#

def complex_sinusoid_inners(N, f1):
    f2 = np.linspace(0, 1, N * N, endpoint=False)
    a = complex_sinusoid(N, f1)
    B = complex_sinusoids(N, f2)
    return np.abs(inner(a, B))

# The math needs to be independent of N so we might as
# well stick to small N in this brute force search
zero_test = complex_sinusoid_inners(8, 0)

# lit skip

plt.plot(np.linspace(0, 1, len(zero_test)), zero_test)
plt.title('inner products of sinusoids ($f_1=0$)')
plt.xlabel('$f_2$ (cycles/sample)')
plt.ylabel('|inner product|')
plt.savefig('zero-test.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# We can see that the inner product of the `0`-frequency sinusoid vector with itself (its length) is `1` as expected and it approaches `1` again as we approach its alias vector at frequency `1` cycles per sample.
# We can also see it approaching `0` at regular intervals and we can get a clue for why by
#

# exp(2jπ0n) = exp(0) = 1 for all n
assert np.allclose(1+0j, complex_sinusoid(N, 0) * np.sqrt(N))

# lit execute
# lit text
#
# Footnotes
# ---------
#
# <p id="footnote-1">Footnote [1] (<a href="#footnote-1-ref">back</a>)</p>
#
# The routine `idft(X)` computes a linear combination of the columns of `A` given by the weights in `X` and is implemented as a matrix multiplication.
# If you're used to thinking of matrix multiplication results in an entry-by-entry way as row-column dot products, it is worth training your mind to also view the results in a column-by-column or row-by-row way as linear combinations of the matrix columns or rows depending on whether the matrix in question is on the left or right side of the expression.
# The matrix or vector on the other side of the expression contains the weights of the combinations in its rows or columns depending on whether *it* appears on the left or right side of the expression.
# Eli Bendersky has a helpful visualization of these operations on his site [here](https://eli.thegreenplace.net/2015/visualizing-matrix-multiplication-as-a-linear-combination/).
# 
#
# <p id="footnote-2">Footnote [2] (<a href="#footnote-2-ref">back</a>)</p>
#
# To restate the claim of the DFT, we claim that `idft(X)` can generate every possible `x` in our `N`-dimensional vector space by combining the columns of `A` with a unique weight vector `X`.
# In linear algebra terms, we claim that the _column space_ of `A` is equivalent to our `N`-dimensional vector space, i.e. that the columns of `A` form a basis for the space.
# Although out of scope for this article, it is not too difficult to prove that every basis for an `N`-dimensional vector space has exactly `N` vectors.
# This [video](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/proof-any-subspace-basis-has-same-number-of-elements) from Kahn Academy can guide you most of the way by proving that every basis of a subspace must contain the same number of vectors (you can then use the columns of the appropriately sized identity matrix as an example basis for any subspace to complete the proof).
# Because the columns of `A` form a basis for an `N`-dimensional space, `A` is invertible and may be called a _change of basis_ matrix that maps coordinate vectors between the standard basis and the basis formed by its columns.
