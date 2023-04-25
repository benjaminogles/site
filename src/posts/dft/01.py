
# lit text
#
# Discrete Fourier Transform
# ==========================
#
# **Published: 2023/04/25**
#
# In this post, I derive the Discrete Fourier Transform (DFT) and show its relationship to convolution.
#
# Derivation
# ----------
#
# Mathematically, a sequence of complex samples, such as the random one below, can be considered a _vector_.
# This vector belongs to a _vector space_ that contains all length-`N` complex vectors.
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
# The claim of the DFT is that we can construct this vector `x`, and any other vector in this vector space, as a unique weighted sum (called a linear combination) of pure sinusoids.
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
# Note: see this [post](/posts/complex-signals/) for an overview of using complex numbers to represent sinusoidal signals.
#
# A set of `K` complex sinusoid vectors with length `N` may form the columns of an `NxK` matrix.
#

def complex_sinusoids(N, freqs):
    """
    Return an `NxK` matrix where `K = len(freqs)`. Each column
    vector contains `N` complex samples of a sinusoid oscillating
    at the associated frequency in `freqs` (cycles/sample).
    """
    return np.column_stack([complex_sinusoid(N, f) for f in freqs])

# lit text
#
# Assume we can define the routine `dft(x)` to compute the weights of `x` as a linear combination of the columns of an `NxK` matrix `Q` where each column contains `N` complex samples of a pure sinusoid (we will see later that `K` must equal `N`).
# Then the inverse DFT (computing `x` from `dft(x)`), is trivially derived as simply computing that linear combination <a id="footnote-1-ref" href="#footnote-1">[1]</a>.
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
    Q = complex_sinusoids(N, dft_freqs(N))
    return np.matmul(Q, X)

# lit text
#
# We can now use `idft(X)` to derive the expression for `dft(x)`.
# We want
#
# `x = idft(dft(x))`
#
# and expanding the definition of `idft(X)` gives
#
# `x = (Q)dft(x)`.
#
# Left multiplying by the inverse of `Q` on both sides completes the derivation:
#
# `(Q^-1)x = dft(x)`.
#
# Note that `(Q^-1)` must be a left and right inverse of `Q` (i.e. `Q` must be invertible) because we need to left multiply the above equation by `Q` to get back to the expression for `idft(x)`.
# So we have our answer on why `K` must be equal to `N` <a id="footnote-2-ref" href="footnote-2">[2]</a>: `Q` must be a square matrix to be invertible and it already has `N` rows corresponding to the `N` samples in `x`.
#
# To help us find the right `dft_freqs(N)`, we will constrain `Q` to have one other important property.
# We want the DFT to be a _unitary transform_, meaning that the magnitudes of `x` and `X` are equal if `X` is the DFT of `x`.
# To state this more generally, we want `Q` to preserve inner products (the inner product of a vector with itself is equal to its magnitude squared).
#

def inner(a, b):
    "Return the inner product of vectors `a` and `b`"
    return np.dot(a.conj(), b)

# lit text
#
# So we need
#
# `inner(a, b) = inner(Qa, Qb)`.
#
# Expanding the definition of `inner` gives
#
# `a*b = (Qa)*(Qb)`
#
# where `*` denotes the conjugate transpose operation and vectors are considered column vectors.
# We can then use the fact that the conjugate transpose of a matrix product is equal to the flipped product of its member's conjugate transposes to write
#
# `a*b = a*Q*Qb`.
#
# For this to be true, we need
#
# `Q*Q = I`.
#
# In other words, we need the inverse of `Q` to be equal to its own conjugate transpose.
# Every entry of `Q*Q` is equal to the inner product of two column vectors in `Q` (rows in `Q*`).
# To make this result equal the identity matrix, we need the inner products of distinct columns in `Q` to be zero and the inner product of each column with itself to be one.
# To state this property in formal terms, we need the columns of `Q` to be _orthonormal_, making `Q` a _unitary_ matrix. 
#
# Although it may be obvious at this point, we can quickly prove that the magnitude of a vector is the same as the magnitude of its DFT if the DFT matrix is unitary.
#
# `inner(x, x) = inner(dft(x), dft(x))`
#
# Expanding the definition of `dft(x)` gives
#
# `inner(x, x) = inner((Q^-1)x, (Q^-1)x)`
#
# and we can replace `Q^(-1)` with `Q*` since `Q` is unitary.
#
# `inner(x, x) = inner(Q*x, Q*x)`.
#
# Expanding the definition of `inner` gives
#
# `x*x = (Q*x)*(Q*x)`
#
# and we get a similar expression as we did before when working with `Q` instead of `Q*`
#
# `x*x = x*QQ*x`.
#
# This is true since `QQ* = Q*Q = I` when `Q*` is the inverse of `Q` i.e. when `Q` is unitary.
# And with the inverse of `Q` defined, we can now implement `dft(x)`.
#

def dft(x):
    """
    Return the weights that generate `x` as a linear combination
    of the columns in the DFT matrix.
    """
    N = len(x)
    Q = complex_sinusoids(N, dft_freqs(N))
    return Q.conj().T @ x

# lit text
#
# Now we need to circle back and actually implement `dft_freqs(N)` to return `N` frequencies that generate orthonormal sinusoid vectors.
# We already implemented `complex_sinusoids(N,freqs)` to always return unit vectors.
# We need to determine which set of frequencies generate mutually orthogonal vectors (their inner product is zero).
#
# Recall (or see this [post](/posts/nyquist-frequency/)) that the samples of two sinusoids with the same initial phase will be exactly equal if the difference between their normalized frequencies (cycles per sample) is an integer.
# Two vectors cannot be orthogonal if they contain the same entries so we can immediately limit our search for `N` frequencies to a range of `1` cycles per sample.
#
# The inner product of two vectors is defined as the sum of their element-wise product where the first vector is conjugated.
# Let's look at the general expression for this element-wise product between two unit vectors containing complex samples of pure sinusoids where `f` and `g` are their two normalized frequencies in cycles per sample and `n` is the sample index.
#
# `(1/sqrt(N))exp(-j2πfn)(1/sqrt(N))exp(j2πgn), 0 <= n < N`.
#
# To simplify this expression, let's ignore its magnitude and define the angular frequencies `φ = 2πf` and `θ = 2πg`.
#
# `exp(-jφn)exp(jθn), 0 <= n < N`.
#
# Since the two terms have the same base, we can sum their exponents:
#
# `exp(j(θ-φ)n), 0 <= n < N`.
#
# The element-wise product looks like another pure sinusoid with angular frequency `θ-φ`.
# We need the samples of this sinusoid to sum to zero whenever `θ` does not equal `φ`.
# For good measure, let's look at the real and imaginary components separately.
#
# `cos((θ-φ)n) + jsin((θ-φ)n), 0 <= n < N`.
#
# The sum of these `N` products is a complex number with the sum of a cosine wave in the real part and the sum of a sine wave in the imaginary part.
# This sum can only be zero when `N` samples completes an integer number of cycles of the wave with angular frequency `θ-φ`.
# This is true when `N(θ-φ)` is an integer multiple of `2π` or, equivalently, when `N(g-f)` is an integer.
#
# To summarize, we need to choose `N` frequencies within a range of `1` cycles per sample where the difference of each pair multiplied by `N` is an integer.
# If we choose a start frequency of zero (any other choice would just yield a permutation of the same set of column vectors), these constraints leave us with one choice for implementing `dft_freqs(N)`.
#

def dft_freqs(N):
    """
    Return the `N` frequencies in cycles/sample used to generate
    the DFT matrix.
    """
    return np.arange(N) / N

# lit text
#
# To double check that this implementation meets both of our constraints, we have
#
# `(N-1)/N < 1`
#
# as the range of frequencies returned by `dft_freqs(N)` and
#
# `N(k/N-l/N) = k-l`
#
# where `k` and `l` are integers, as the difference between each pair of frequencies multiplied by `N`.
# With `dft_freqs(N)` implemented, we can check the rest of our program.
#

I = np.identity(N)
Q = complex_sinusoids(N, dft_freqs(N))
assert np.allclose(I, Q.conj().T @ Q)
assert np.allclose(I, Q @ Q.conj().T)

X = dft(x)
assert np.isclose(inner(x, x), inner(X, X))
assert np.allclose(x, idft(X))

# lit execute
# lit text
#
# We can also take a look at the DFT of `x`.
# It's not that interesting to look at but neither was `x`.
# The point is that if we add the `N` sinusoids in the columns of the DFT matrix together, weighted with these magnitudes and rotated to these initial phase angles, we will construct `x`.
#

# lit skip

_, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(np.fft.fftshift(np.fft.fftfreq(N)), np.abs(np.fft.fftshift(X)))
axs[0].set_ylabel('magnitude')
axs[1].plot(np.fft.fftshift(np.fft.fftfreq(N)), np.angle(np.fft.fftshift(X)))
axs[1].set_xlabel('normalized frequency')
axs[1].set_ylabel('phase')
plt.title('X')
plt.savefig('X.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# Note: we plot the right half of the DFT vector on the left side of the plot since the column vectors in `Q` with frequencies in the range `(0.5, 1)` also represent the frequencies in the range `(-0.5, 0)`.
#
# Relationship to Convolution <a id="footnote-3-ref" href="#footnote-3">[3]</a>
# -----------------------------------------------------------------------------
#
# The matrix `Q` shows up again in what initally seems to be an unrelated problem: _simultaneously diagonalizing_ the set of all _circulant matrices_.
#
# **Circulant Matrices**
#
# Circulant matrices are square matrices where each column is a distinct circular rotation of the first column.
# The columns are ordered so that each column is only different from its neighbors by one circular shift.
# The same relationship holds for the rows of a circulant matrix.
#

import scipy.linalg
C = scipy.linalg.circulant([0, 0, 1])

# lit skip

print("C =", np.array2string(C, prefix=" "*3)[1:-1])

# lit unskip
# lit text
#
# `C` is a simple but important example of a circulant matrix.
# It can obviously be extended to any size but the `3x3` example looks like this.
#
# lit execute
#
# Consider the products `aC` or `Ca` depending on whether `a` is a row or column vector.
#

a = np.arange(3)

# lit skip

print('a  =', a)
print('Ca =', C @ a)
print('aC =', a @ C)

# lit unskip
# lit text
#
# In both cases, the result will be a circular rotation of `a`.
#
# lit execute
#
# This is the simplest form of the operation implemented by circulant matrices: _circular convolution_.
# The matrix `C` is simply picking out an entry in `x` for each position in the result by rotating the location of a `1` in its columns.
# A general circulant matrix `M` will contain other non-zero entries that can be seen as weights for combining the entries of `x` in `Mx`.
# Each position of the result will contain a weighted sum of the entries of `x` but with the weights centered at a different position in `x`.
# The first column of the circulant matrix, and the vector `x` can be extended with trailing zeros to avoid the "edge effects" of circular convolution and instead implement _linear convolution_.
#
# **Diagonalization**
#
# Diagonalizing the matrix `C` (or any other square matrix) means finding matrices `P` and `D` such that
#
# `C = (P)D(P^-1)`
#
# where `D` is a diagonal matrix (zero everywhere except the main diagonal).
# This problem can also be described as finding the _eigenvalues_ (the diagonal entries of `D`) and _eigenvectors_ (columns of `P`) of `C`.
# To see why, consider an eigenvector `v` of `C`.
# By definition, we have
#
# `Cv = λv`
#
# for some scalar `λ`, called an eigenvalue.
# And if we can diagonalize `C`, then we can rewrite the equation above as
#
# `(P)D(P^-1)v = λv`
#
# which we can then rearrange to
#
# `D(P^-1)v = λ(P^-1)v`
#
# meaning that `λ` is also an eigenvalue of `D`, just with a different eigenvector: `(P^-1)v`.
# Since `D` is diagonal, its eigenvalues must be equal to its diagonal entries, which shows that `D` must contain the eigenvalues of `C`.
#
# Diagonal matrices are useful because they reduce matrix multiplication to an independent scaling of rows or columns.
#
# **Diagonalization of `C`**
#
# We will now find the eigenvalues and eigenvectors of an `NxN` matrix `C` with the same form as the `3x3` version shown above (the permutation matrix).
# We need `N` non-zero eigenvalues `λ_k` and eigenvectors `v_k`:
#
# `C(v_k) = (λ_k)(v_k), 0 <= k < N`.
#
# We know that `C` is just going to rotate the entries of `v_k` one step.
# To satisfy the above equation, we need successive entries of `v_k` to be related by `λ_k`.
#
# `v_k[(n+1)%N] = (λ_k)(v_k[n]), 0 <= k,n < N`.
#
# We can generalize this to a statement relating entries in `v_k` separated by `m` steps:
#
# `v_k[(n+m)%N] = ((λ_k)^m)(v_k[n]), 0 <= k,n < N`.
#
# If `m = N`, this simplifies to
#
# `v_k[n] = ((λ_k)^N)(v_k[n]), 0 <= n < N`
#
# meaning that `(λ_k)^N = 1`.
# There is a name for numbers that reduce to `1` when raised to the `N`th power.
# They are called `N`th roots of unity and there are exactly `N` distinct complex `N`th roots of unity:
#
# `exp(j2πk/N), 0 <= k < N`.
#
# These are necessarily equal to the `N` eigenvalues of `C`.
# Now we can solve for the entries of `v_k` using the previously stated relationship:
#
# `v_k[(n+m)%N] = ((λ_k)^m)(v_k[n]), 0 <= k,n < N`.
#
# The scale of eigenvectors is arbitrary so we can assume `v_k[0]` is `1` and solve for the rest of the entries of `v_k` relative to `v_k[0]`.
#
# `v_k[m] = (λ_k)^m, 0 <= k,m < N`.
#
# So the entries of the `k`th eigenvector are given by
#
# `v_k[m] = exp(j2πkm/N), 0 <= k,m < N`.
#
# This is the same expression used to generate the `k`th column of our DFT matrix.
# So we have proven that the eigenvectors of `C` are equal to the columns of `Q`.
#
# **Simultaneous Diagonalization of all Circulant Matrices**
#
# Now we would like to prove that all circulant matrices share this same set of eigenvectors.
# This is called simultaneous diagonalization.
# For an arbitrary `NxN` circulant matrix `B` we need
#
# `B(v_k) = (γ_k)(v_k), 0 <= k < N`
#
# for some set of eigenvalues `γ_k`.
# Multiply this expression by `C` on both sides to get
#
# `CB(v_k) = (γ_k)C(v_k), 0 <= k < N`
#
# and if we assume (for now) that `C` and `B` commute, then we have
#
# `BC(v_k) = (γ_k)C(v_k), 0 <= k < N`
#
# and because `v_k` is an eigenvector of `C`, we can simplify this to
#
# `B(λ_k)(v_k) = (γ_k)(λ_k)(v_k), 0 <= k < N`
#
# showing that `(λ_k)(v_k)`, and therefore `v_k` (the scale of an eigenvector is irrelevant), is an eigenvector of `B`.
#
# ** `C` Commutes with all Circulant Matrices**
#
# Our approach to the simultaneous diagonalization of all circulant matrices relied on an assumption that `CB = BC` for any `NxN` circulant matrix `B`.
# We can quickly convince ourselves of this fact by looking at a representative `3x3` example.
#

B = scipy.linalg.circulant([1, 2, 3])
CB = C @ B
BC = B @ C

# lit skip

print("B  =", np.array2string(B, prefix=" "*4)[1:-1])
print()
print("CB =", np.array2string(CB, prefix=" "*4)[1:-1])
print()
print("BC =", np.array2string(BC, prefix=" "*4)[1:-1])

# lit unskip
# lit text
# lit execute
#
# Notice that the entries of a circulant matrix are constant along the diagonals.
# For any circulant matrix, the number to the left of each entry will be the same as the number below it (wrapping around the ends on the first column and last row).
# This means that rotating the rows of `B` upwards (the effect of `CB`) and rotating the columns of `B` rightwards (the effect of `BC`) will yield the same result.
#
# **Summary**
#
# We have proven that `NxN` circulant matrices share a set of eigenvectors equal to the columns of the `NxN` DFT matrix `Q`.
# This means that we can write any `NxN` circulant matrix `B` in terms of `Q`
#
# `B = QDQ*`.
#
# Consider the first column `b` of `B`.
# In the above expression, it is computed as a linear combination of the columns of `Q` given by the weights in the first column of `DQ*`.
# And the first column of `DQ*` simply contains the diagonal entries of `D`.
# To see why, recall that the first entry of every row in `Q*` (conjugate of columns in `Q`) is equal to
#
# `exp(-1j*k*0/N) = exp(0) = 1, 0 <= k < N`.
#
# and `DQ*` will scale each row in `Q*` by the diagonal entries of `D`.
# This means that the first column `b` of `B` is a linear combination of the columns of `Q` given by the weights `d` on the diagonal of `D`:
#
# `b = Qd`.
#
# Which means we can also write the digaonal entries of `D` in terms of `b`
#
# `d = Q*b`.
#
# Which is all to say that the eigenvalues of `B` (the diagonal entries of `D`) are easily computed as the DFT of the first column in `B`.
# This means that any product `Bx` can be written as
#
# `Bx = QDQ*x = idft(dft(b)∘dft(x))`
#
# where `b` is the first column of `B` and `∘` denotes element-wise multiplication.
# In other words, convolution with `b` can be implemented by matrix multiplication with `B` or by element-wise multiplication wth `dft(b)` through DFT operations.
#
# Footnotes
# ---------
#
# <p id="footnote-1">Footnote [1] (<a href="#footnote-1-ref">back</a>)</p>
#
# The routine `idft(X)` computes a linear combination of the columns of `Q` given by the weights in `X` and is implemented as a matrix multiplication.
# If you're used to thinking of matrix multiplication results in an entry-by-entry way as row-column dot products, it is worth training your mind to also view the results in a column-by-column or row-by-row way as linear combinations of the matrix columns or rows depending on whether the matrix in question is on the left or right side of the expression.
# The matrix or vector on the other side of the expression contains the weights of the combinations in its rows or columns depending on whether *it* appears on the left or right side of the expression.
# Eli Bendersky has a helpful visualization of these operations on his site [here](https://eli.thegreenplace.net/2015/visualizing-matrix-multiplication-as-a-linear-combination/).
# 
#
# <p id="footnote-2">Footnote [2] (<a href="#footnote-2-ref">back</a>)</p>
#
# As is often the case in linear algebra, there is more than one way to see that `Q` must be an `NxN` matrix but they are all equivalent to asserting that `Q` is an invertible matrix.
# To restate the claim of the DFT, we claim that `idft(X)` can generate every possible `x` in our `N`-dimensional vector space by combining the columns of `Q` with a unique weight vector `X`.
# In linear algebra terms, we claim that the _column space_ of `Q` is equivalent to our `N`-dimensional vector space, i.e. that the columns of `Q` form a basis for the space.
# Although out of scope for this article, it is not too difficult to prove that every basis for an `N`-dimensional vector space has exactly `N` vectors.
# This [video](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/proof-any-subspace-basis-has-same-number-of-elements) from Kahn Academy can guide you most of the way by proving that every basis of a subspace must contain the same number of vectors (you can then use the columns of the appropriately sized identity matrix as an example basis for any subspace to complete the proof).
#
# <p id="footnote-3">Footnote [3] (<a href="#footnote-3-ref">back</a>)</p>
#
# I learned about this topic from a great [article](https://arxiv.org/abs/1805.05533) written by Bassam Bamieh.
# Look there for more detail.
#
