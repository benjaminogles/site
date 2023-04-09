
# lit text
#
# Discrete Fourier Transform
# ==========================
#
# In this post, I derive the Discrete Fourier Transform (DFT) as a matrix and then show a few other ways to interpret its meaning.
#
# Matrix Derivation
# -----------------
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
# The claim of the DFT is that we can construct this vector `x`, and any other such vector `x`, as a unique weighted sum (called a linear combination) of pure sinusoids.
# The weights assigned to each sinusoid give us an indication of the frequency content in `x`.
# I wouldn't say this claim is obviously true, just by looking at our example `x`, so it is worthy of a thorough derivation.
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
# Assume we can define the routine `dft(x)` to compute the weights of `x` as a linear combination of the columns of an `NxK` matrix `A` where each column contains `N` complex samples of a pure sinusoid (we will see later that `K` must equal `N`).
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
    A = complex_sinusoids(N, dft_freqs(N))
    return np.matmul(A, X)

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
# So we have our answer on why `K` must be equal to `N` <a id="footnote-2-ref" href="footnote-2">[2]</a>: `A` must be a square matrix to be invertible and it already has `N` rows corresponding to the `N` samples in `x`.
#
# To help us find the right `dft_freqs(N)` to construct the right invertible `A`, we will constrain `A` to have one other important property.
# We want the DFT to be a _unitary transform_, meaning that the magnitudes of `x` and `X` are equal if `X` is the DFT of `x`.
# To state this more generally, we will want `A` to preserve inner products (the inner product of a vector with itself is equal to its magnitude squared).
#

def inner(a, b):
    "Return the inner product of vectors `a` and `b`"
    return np.dot(a.conj(), b)

# lit text
#
# So we need
#
# `inner(a, b) = inner(Aa, Ab)`.
#
# Expanding the definition of `inner` gives
#
# `a*b = (Aa)*(Ab)`
#
# where `*` denotes the conjugate transpose operation and vectors are considered column vectors.
# We can then use the fact that the conjugate transpose of a matrix product is equal to the flipped product of its member's conjugate transposes to write
#
# `a*b = a*A*Ab`.
#
# For this to be true, we need
#
# `A*A = I`.
#
# In other words, we need the inverse of `A` to be equal to its own conjugate transpose.
# Every entry of `A*A` is equal to the inner product of two column vectors in `A` (rows in `A*`).
# To make this result equal the identity matrix, we need the inner products of distinct columns in `A` to be zero and the inner product of each column with itself to be one.
# To state this property in formal terms, we need the columns of `A` to be _orthonormal_, making `A` a _unitary_ matrix. 
#
# Although it may be obvious at this point, we can quickly prove that the inner product of a vector is the same as the inner prodcut of its DFT.
#
# `inner(x, x) = inner(dft(x), dft(x))`
#
# Expanding the definition of `dft(x)` gives
#
# `inner(x, x) = inner((A^-1)x, (A^-1)x)`
#
# and we can replace `A^(-1)` with `A*` since `A` is unitary.
#
# `inner(x, x) = inner(A*x, A*x)`.
#
# Expanding the definition of `inner` gives
#
# `x*x = (A*x)*(A*x)`
#
# and we get a similar expression as we did before when working with `A` instead of `A*`
#
# `x*x = x*AA*x`.
#
# This is true since `AA* = A*A = I` when `A*` is the inverse of `A`.
# And with the inverse of `A` defined, we can now implement `dft(x)`.
#

def dft(x):
    """
    Return the weights that generate `x` as a linear combination
    of the columns in the DFT matrix.
    """
    N = len(x)
    A = complex_sinusoids(N, dft_freqs(N))
    return A.conj().T @ x

# lit text
#
# Now we need to circle back and actually implement `dft_freqs(N)` to return `N` frequencies that generate orthonormal sinusoid vectors.
# We already implemented `complex_sinusoids(N,freqs)` to always return unit vectors.
# We need to determine which set of frequencies generate mutually orthogonal vectors (their inner product is zero).
#
# Recall (or see this [post](/posts/nyquist-frequency/)) that the samples of two sinusoids with the same initial phase will be exactly equal if the difference between their normalized frequencies (cycles per sample) is an integer.
# Two vectors cannot be orthogonal if they contain the same entries so we can immediately limit our search for `N` frequencies to a range of `1` cycles per sample.
#
# The inner product of two vectors is defined as the sum of their point-wise product where the first vector is conjugated.
# Let's look at the general expression for this point-wise product between two unit vectors containing complex samples of pure sinusoids where `f` and `g` are their two normalized frequencies in cycles per sample and `n` is the sample index.
#
# `(1/sqrt(N))exp(-j2πfn)(1/sqrt(N))exp(j2πgn)`,  `0 <= n < N`.
#
# To simplify this expression, let's ignore its magnitude and define the angular frequencies `φ = 2πf` and `θ = 2πg`.
#
# `exp(-jφn)exp(jθn)`,  `0 <= n < N`.
#
# Since the two terms have the same base, we can sum their exponents:
#
# `exp(j(θ-φ)n)`, `0 <= n < N`.
#
# The point-wise product terms just look like another pure sinusoid with angular frequency `θ-φ`.
# We need these products to sum to zero whenever `θ` does not equal `φ`.
# For good measure, let's look at the real and imaginary components separately.
#
# `cos((θ-φ)n) + jsin((θ-φ)n)`, `0 <= n < N`.
#
# The sum of these point-wise product terms is a complex number with the sum of a cosine wave in the real part and the sum of a sine wave in the imaginary part.
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
# `N(k/N-l/N) = k-l`, `k` and `l` integers
#
# as the difference between each pair of frequencies multiplied by `N`.
# With `dft_freqs(N)` implemented, we can check the rest of our program.
#

I = np.identity(N)
A = complex_sinusoids(N, dft_freqs(N))
assert np.allclose(I, A.conj().T @ A)
assert np.allclose(I, A @ A.conj().T)

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
# Note: we plot the right half of the DFT vector on the left side of the plot since the column vectors in `A` with frequencies in the range `(0.5, 1)` also represent the frequencies in the range `(-0.5, 0)`.
#
# The DFT as an Eigen Basis <a id="footnote-3-ref" href="#footnote-3">[3]</a>
# --------------------------------------------------------------------------
#
# The matrix `A` shows up again in what initally seems to be an unrelated problem: _simultaneously diagonalizing_ the set of all _circulant matrices_.
#
# Circulant Matrices
# ******************
#
# Circulant matrices are square matrices where each column is a distinct circular rotation of the first column.
# The columns are ordered so that each column is only different from its neighbors by one circular shift.
# The same type of pattern can be seen along the rows of a circulant matrix.
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
# Diagonalization
# ***************
#
# Diagonalizing the matrix `C` (or any other square matrix) means finding matrices `P` and `D` such that
#
# `C = (P)D(P^-1)`
#
# where `D` is a diagonal matrix (zero everywhere except the main diagonal).
# This problem can also be described as finding the _eigenvalues_ (the diagonal entries of `D`) and _eigenvectors_ (columns of `P`) of `C`.
# To see why, consider an eigenvector `v` of `C`.
# Then, by definition
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
# Diagonalization of `C`
# **********************
#
# We will now find the eigenvalues and eigenvectors of an `NxN` matrix `C` with the same form as the `3x3` version shown above (the permutation matrix).
# We need `N` non-zero eigenvalues `λ` and eigenvectors `v` where
#
# `Cv = λv`.
#
# We know that `C` is just going to rotate the entries of `v` one step.
# To satisfy the above equation, we need successive entries of `v` to be related by `λ`.
#
# `v[(n+1)%N] = λv[n]`, `0 <= n < N`.
#
# We can generalize this to a statement relating entries in `v` separated by `m` steps:
#
# `v[(n+m)%N] = (λ^m)v[n]`, `0 <= n < N`.
#
# If `m = N`, this simplifies to
#
# `v[n] = (λ^N)v[n]`, `0 <= n < N`
#
# meaning that `λ^N = 1`.
# There is a name for numbers that reduce to `1` when raised to the `N`th power.
# They are called `N`th roots of unity and there are exactly `N` distinct complex `N`th roots of unity:
#
# `exp(j2πn/N)`, `0 <= n < N`.
#
# These are the `N` eigenvalues of `C`.
# Now we can solve for the entries of `v` using the previous relationship:
#
# `v[(n+m)%N] = (λ^m)v[n]`, `0 <= n < N`.
#
# The absolute scale of eigenvectors is arbitrary so we can assume `v[0]` is `1` and solve for the rest of the entries of `v` relative to `v[0]`.
#
# `v[m] = λ^m`, `0 <= m < N`.
#
# So the entries of the `k`th eigenvector are given by
#
# `v_k[m] = exp(j2πkm/N)`, `0 <= k,m < N`.
#
# where we have paired the `k`th eigenvector with the `k`th complex `N`th root of unity as its eigenvalue `λ`.
# This is the same expression used to generate the `k`th column of our DFT matrix.
# So we have proven that the eigenvectors of `C` are equal to the columns of `A`.
#
# Simultaneous Diagonalization of Circulant Matrices
# **************************************************
#
# We would
# This proves that the eigenvectors of this particular `C` are equal to the columns of the DFT matrix but we still need to prove that this set of eigenvectors is shared by all circulant matrices.
# For some other arbitrary `NxN` circulant matrix `B`, we want
#
# `B(v_m) = γ(v_m)`, `0 <= m < N`
#
# Multiply this expression by `C` on both sides to get
#
# `CB(v_m) = γC(v_m)`, `0 <= m < N`
#
# and if we assume (for now) that `C` and `B` commute, we have
#
# `BC(v_m) = γC(v_m)`, `0 <= m < N`
#
# showing that `C(v_m)` is an eigenvector of `B`.
# But since `v_m` is an eigenvector of `C`, we know that `C(v_m)` is just a scalar multiple of `v_m`.
# Eigenvectors are only unique up to scalar multiples so we have proved that `v_m` is an eigenvector of `B`, with some eigenvalue `γ`, if `C` and `B` commute.
#
# We can convince ourselves (see the previously mentioned article for a relatively simple proof)  that `C` must commute with any `NxN` circulant matrix `B` by visually examining the `3x3` case for a simple `B`.
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
#
# Recall that `CB` and `BC` are just circular rotations of the rows and columns in `B` and that each row and column in `B` contains the same entries, circularly rotated to different positions.
# In this case, those entries are `[1, 2, 3]` which stand in as labels on the entries of a general circulant matrix `B`.
#
# lit execute
#
# Since `C` commutes with any `NxN` circulant matrix `B` <a id="footnote-4-ref" href="#footnote-4">[3]</a>, we have proved that every `B` shares the same set of eigenvectors given by the columns of the DFT matrix.
# This means any convolution given by a circulant matrix `B` can be written as
#
# `Bx = ADA*x = idft((D)dft(x))`
#
# where
#
# `D = A*BA`
#
# by definition.
# We can simplify the expression for `D` as the DFT of the kernel vector that generates `B`.
#
# Eigen Basis
# ***********
#
# The columns of an `NxN` invertible matrix form a _basis_ for an `N`-dimensional vector space.
# This just gives a name to the concept that we covered when deriving the DFT matrix.
# If
#
# `x = Py`
#
# constructs `x` as a linear combination of the columns in `P`, then
#
# `(P^-1)x = y`
#
# computes the weights `y`.
# This means that we can construct any `x` as a unique linear combination of the columns of `P` with the weights given by `(P^-1)x`.
# Because `P` is made up of eigenvectors, the basis formed by its columns may be called an `eigenbasis`.
#
# Any matrix product with a diagonalizable matrix `C` can be written in terms of a diagonal matrix product surrounded by _change of basis_ operations.
#
# `Cx = (P)D(P^-1)x`
#
# By definition, if we can diagonalize `C`, then we have a set of eigenvectors that form the columns of an invertible matrix `P`, and equialently a _basis_ (in this case, called an _eigenbasis_) over the corresponding vector space.
# This just gives a name to the already familiar idea that we can find unique weights to compute any vector as a linear combination of the eigenvectors of `C`.
# Computing those weights for a particular vector `x` is as simple as computing `(P^-1)x` since
#
# `(P)(P^-1)x = x`
#
# can be seen as giving the linear combination of the columns of `P` with weights in `(P^-1)x`.
# This is of course the same line of reasoning we used to derive the DFT and indeed the DFT is also a change of basis onto the basis formed by the columns of `A`.
# Going back to the matrix product `Cx` where `C` is diagonalizable:
#
# `Cx = (P)D(P^-1)x`.
#
# From right to left, we can interpret this expression as a change of basis to the eigenbasis of `C`, followed by a scaling of the weights for `x` on that basis by the eigenvalues of `C`, followed by "undoing" the change of basis by computing the resulting linear combination of the columns of `P`.
#
# Simultaneously diagonalizing the entire set of circulant matrices means finding one set of eigenvectors `P` shared by all circulant matrices.
# The result is one eigenbasis that reduces circular convolution to change of basis and scaling operations.
# This result may not sound useful, trading one matrix multiplicaton for three others, but it ends up being _very_ useful for at least three reasons.
#
# 1. The shared eigenbasis allows us to directly compare the effects of circulant matrices i.e. we can directly compare the eigenvalues of circulant matrices
# 2. For circulant matrices, we will see that `P` is equal to the DFT matrix `A` and we often want to compute the DFT of signal vectors anyway, so we get to reuse this computation for convolution
# 3. The structure of `A` gives way to efficient algorithms for computing the change of basis operations (DFT and IDFT)
#
# After all this introduction, we still need to prove that all circulant matrices share the eigenvectors given by the columns of the DFT matrix `A`.
#
# As a Bank of Bandpass FIR Filters
# ---------------------------------
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
# As is often the case in linear algebra, there is more than one way to see that `A` must be an `NxN` matrix but they are all equivalent to asserting that `A` is an invertible matrix.
# To restate the claim of the DFT, we claim that `idft(X)` can generate every possible `x` in our `N`-dimensional vector space by combining the columns of `A` with a unique weight vector `X`.
# In linear algebra terms, we claim that the _column space_ of `A` is equivalent to our `N`-dimensional vector space, i.e. that the columns of `A` form a basis for the space.
# Although out of scope for this article, it is not too difficult to prove that every basis for an `N`-dimensional vector space has exactly `N` vectors.
# This [video](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/proof-any-subspace-basis-has-same-number-of-elements) from Kahn Academy can guide you most of the way by proving that every basis of a subspace must contain the same number of vectors (you can then use the columns of the appropriately sized identity matrix as an example basis for any subspace to complete the proof).
#
# <p id="footnote-3">Footnote [3] (<a href="#footnote-3-ref">back</a>)</p>
#
# I learned about this topic from a great [article](https://arxiv.org/abs/1805.05533) written by Bassam Bamieh.
# Look there for more detail.
#
# <p id="footnote-4">Footnote [4] (<a href="#footnote-4-ref">back</a>)</p>
#
# Actually all circulant matrices must mutually commute as we can show this statement's equivalence to the point of interest: that they share the same set of eigenvectors.
# First, write the product of two arbitrary circulant matrices `CB` in terms of their diagonalized forms with this shared set of eigenvectors.
#
# `CB = (P)D(P^-1)(P)E(P^-1)`
#
# where `D` and `E` contain the eigenvalues of `C` and `B` respectively.
# Now simplify the middle term.
#
# `CB = (P)DE(P^-1)`
#
# Diagonal matrices always commute, so we have
#
# `CB = (P)ED(P^-1)`
#
# which we can now rewrite in terms of `C` and `B` by bringing back that middle term equal to the identity matrix
#
# `CB = (P)E(P^-1)(P)D(P^-1) = BC`.
