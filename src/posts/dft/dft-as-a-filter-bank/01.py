
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
# A fair bit of signal processing background is covered along the way, including sampling, complex numbers and linear time-invariant systems.
#
# Example Problem
# ---------------
#
# We will generate several arrays of complex samples, each one representing a pure sinusoid oscillating at a unique, constant frequency.
# We will then combine these arrays by adding aligned samples together.
# Our task will be to estimate the individual frequency of each wave by analyzing the combined signal.
#

# Seed random number generator
import numpy as np
rng = np.random.default_rng(seed=184027)
# Pick number of waves to sample
nwaves = 4
# Pick number of samples to generate
nsamples = 128
# Sample indices
t = np.arange(nsamples)
# Bound cycle length in samples for smoother plots
min_wavelength_samps = 10
# Pick random frequency for each wave (cycles/sample)
min_freq = -1.0 / min_wavelength_samps
max_freq = 1.0 / min_wavelength_samps
freqs = rng.uniform(min_freq, max_freq, nwaves)
# Generate samples for each wave
waves = [np.exp(2j * np.pi * f * t) for f in freqs]
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
# Background - Sampling
# ---------------------
#
# If you are used to frequency in Hertz, note that I've simply factored out physical units by setting our sample spacing equal to 1 so that sample index can be used as a proxy for e.g. time.
# In an actual application, you could scale our normalized frequency (cycles/sample) by the sampling rate (samples/second) to get back to Hertz (cycles/second).
#
# I have biased the generation of the example problem towards middle frequencies because more samples per cycle makes some of the plots look nicer.
# Per Nyquist, the actual bounds in frequency that our complex samples can accurately represent is `(-0.5, 0.5)`.
#
# The notion of negative frequency may be confusing.
# As the derivative of phase, frequency is negative when phase (`2πft` above) is decreasing sample to sample.
# One cycle is defined as `2π` radians no matter which direction phase is advancing (a clockwise path around a circle is the same length as a counter-clockwise path).
# On a more practical level, I will show later on how you can shift the phases of complex samples to center an arbitrary frequency at the zero cycles/sample level.
# So you can analyze a set of positive frequencies with some relationship to the physical world as a set of negative and positive frequencies centered around zero.
#
# In case the term "Nyquist" is unfamiliar: Harry Nyquist is credited for formalizing the limits of representing a continuous time signal as equally spaced samples of that signal.
# The frequencies `-0.5` and `0.5` cycles/sample are called Nyquist frequencies because they are the first frequencies that we cannot unambiguously represent as discrete samples as we move outward from zero.
# Imagine generating samples starting with a phase of `0` radians and either increasing or decreasing this phase by `π` radians (half a cycle) at each sample.
# Both frequencies will visit the same set of discrete phases in lock step, yielding the exact same sample sequence.
#
# Similarly, frequencies beyond the Nyquist limits will always generate the same set of samples as a frequency within the `(-0.5, 0.5)` range.
# As one more concrete example, consider the frequencies `-0.1` and `0.9` cycles/sample.
# Why would a rate of `0.9 * 2π` radians/sample visit the same set of phases as `-0.1 * 2π` radians/sample?
# Again, it is easiest to understand by relating this changing phase to a path around a circle.
# Traveling most of the way around a circle in a counter-clockwise direction will always reach the same spot as traveling a proportionally smaller distance in a clockwise direction.
# So `0.9 * 2π` radians may not be the same value as `-0.1 * 2π` radians, but when passed to the cosine or sine functions (which trace a circle when taken together as `(x, y)` pairs), both arguments will yield the same result.
#
# Background - Complex Numbers
# ----------------------------
#
# How can complex samples represent a real-valued sinusoid?
#
# As a quick review, complex numbers take the form `x + jy` where `x` is called the real part, `y` is called the imaginary part and `j*j = -1`.
# If we plot the point `(x, y)` on a 2D plane, it is clear that `x + jy` can be alternately defined in terms of a amplitude `A` (measured from the origin) and a phase angle `θ` (measured from the positive horizontal axis).
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

# Random phases and amplitudes
phases = rng.uniform(0, 2 * np.pi, nsamples)
amplitudes = rng.uniform(-1, 1, nsamples)
# Generate a cosine, sine and complex wave
a = amplitudes * np.cos(phases)
b = amplitudes * np.sin(phases)
c = amplitudes * np.exp(1j * phases)
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
# But it is not contrived solely for this purpose.
# It is also closely related to practical application.
#
# Pure sinusoids can be modulated and transmitted with radio hardware to implement wireless communication systems.
# Multiple transmitters can be active at the same time if they are transmitting at different frequencies with sufficient separation.
# Our example problem simulates the physical phenomenon (called wave superposition) that occurs when waves from multiple transmitters meet in the air.
# The waves combine such that the displacement of the combined wave at every point is equal to the sum of the individual wave displacements.
#
# If you used another set of radio hardware to receive and digitize this combined waveform, you would get something that looks like our example problem.
# You could use the DFT to see which frequencies are active.
# As a sneak peek, here is a zoomed in look at the DFT of our combined signal.
#
# lit skip

S = np.fft.fftshift(np.fft.fft(signal, nsamples*2))/(nsamples*2)
f = np.fft.fftshift(np.fft.fftfreq(nsamples*2))
plt.plot(f[nsamples//2:-nsamples//2], np.real(S * S.conj())[nsamples//2:-nsamples//2])
plt.title('DFT(signal)')
plt.xlabel('frequency (cycles/sample)')
plt.ylabel('mag^2')
plt.savefig('dft.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# Step 1
# ------
#
# With that background out of the way, we can get started on solving this example problem.
# As a first step, we could pick a frequency and try to detect whether a wave at that frequency was summed into our signal.
# The simplest frequency to reason about is zero cycles/sample because its wave is just a sequence of one repeated sample:

phi = rng.uniform() # random phase offset
zero_freq_wave = amplitudes * np.exp(1j * (2 * np.pi * 0 * t + phi))
repeated_samples = amplitudes * np.exp(1j * phi)
assert np.allclose(zero_freq_wave, repeated_samples)

# lit text
#
# When summed into another signal, this wave would simply shift each sample up or down by a constant.
# This means that we can detect a zero-frequency wave in our signal by computing the signal's mean value.
#

# lit skip
assert np.abs(np.mean(signal)) < 0.2, "did the rng seed change?"
# lit unskip

m = np.mean(signal)
print(f'mean={m} mag={abs(m)}')

# lit text
#
# In this case, the mean value of our signal is near `0` in both the real and imaginary parts.
#
# lit execute
#
# If one of our waves was a constant, we would expect the mean value of the signal to reflect that sample and have a larger magnitude (all of our waves have a magnitude of `1`).
# But then shouldn't we also expect a mean value of _exactly_ `0` if all of these waves have non-zero frequency?
#
# It is true that the expected value of every pure sinusoid with non-zero frequency is `0` and that the sum of those waves also has an expected value of `0`.
# If we sampled a whole number of cycles from each wave, then we would get the result we expect.
# But here we have collected an arbitrary number of samples from each wave without considering their frequencies.
# Any partial cycle that we sampled will skew our mean statistic.
# We may not have even collected a full cycle's worth of samples for a wave if its frequency is very close to, but not exactly zero.
#
# All this is to say that our mean detector for a zero-frequency wave is not perfect.
# And we can characterize excactly how imperfect it is by running some quick tests.
#

# Divide wavelengths by multiple of window length
tests = np.linspace(0, 0.5, nsamples * 5)
# Test frequencies in (-0.5, 0.5)
tests = np.concatenate((-tests[:-1][::-1], tests))
# Compute mean value of each with a window of nsamples
results = np.mean([np.exp(2j*np.pi*f*t) for f in tests], axis=-1)

# lit skip
plt.plot(tests, np.abs(results))
plt.title(f'mean of {nsamples} samples')
plt.xlabel('frequency')
plt.ylabel('mag')
plt.savefig('mean-detector.png')
plt.close()

# lit unskip
# lit text
# lit execute
#
# You may wonder if these test results are directly applicable to our example problem.
# Here I applied our detector to a pure sinusoid in each test.
# In our example problem, I applied the detector to the sum of several pure sinusoids.
# How do we characterize the behavior of our detector for near-zero-frequency waves when the input is more complicated like this?
#
# Background - Linear Time-Invariant (LTI) Systems
# ------------------------------------------------
#
# Computing the mean value of a signal in blocks like this can be considered a linear time-invariant system if it meets the following conditions.
#

# Linear (scaling and adding inputs is the same as scaling and adding outputs)
a = rng.uniform()
b = rng.uniform(size=nsamples)
c = rng.uniform()
d = rng.uniform(size=nsamples)
assert np.isclose(np.mean(a * b + c * d), a * np.mean(b) + c * np.mean(d))
# Time Invariant (delaying the input just delays the same output)
pass

# lit execute
# lit text
#
# I couldn't think of nice way to distill the time-invariance property down to a concrete example.
#
# Back to Step 1
# --------------
#
# Let's zoom in on the middle frequencies of our test results for the mean value of pure sinusoids in a finite block of samples.
#
# lit skip

plt.plot(tests[len(tests)//2-50:len(tests)//2+50], np.abs(results[len(results)//2-50:len(results)//2+50]))
plt.vlines([-1.0/nsamples, 1.0/nsamples], 0, 0.5, color='tab:green')
plt.text(1.0/nsamples, 0.5, f'1/{nsamples}')
plt.vlines([-2.0/nsamples, 2.0/nsamples], 0, 0.5, color='tab:red')
plt.text(2.0/nsamples, 0.5, f'2/{nsamples}')
plt.vlines([-3.0/nsamples, 3.0/nsamples], 0, 0.5, color='tab:brown')
plt.text(3.0/nsamples, 0.5, f'3/{nsamples}')
plt.title(f'mean of {nsamples} samples')
plt.xlabel('frequency')
plt.ylabel('mag')
plt.savefig('mean-detector-zoomed.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# As expected, the zero-frequency wave has the largest mean value of `1`.
# I've marked the first few frequencies in either direction that have a mean value of `0`.
# Unsurprisingly, they are frequencies that complete a whole number of cycles (`1`, `2` or `3`) in the number of samples we collected.
# Between these are frequencies that include a partial cycle within the sampled time range.
#
# We see small peaks right in the middle of the zero-mean frequencies because the cosine and sine functions cross zero halfway through a cycle, so the last half of a cycle balances out the first half of the cycle.
# The peaks are shorter as we move away from `0` because cycles are shorter (wavelength) as the absolute value of frequency increases.
#
# Frequencies very close to zero will not come close to completing even a single cycle in this many samples.
# These waves have a mean value close to `1`.
# So our detector for the zero-frequency wave is actually more of a detector for near-zero-frequency waves.
# We can squeeze this range of near-zero-frequency waves by analyzing more samples at once.

# Collect more samples
t2 = np.arange(nsamples * 2)
# Divide wavelengths by multiple of window length
tests = np.linspace(0, 0.5, nsamples * 2 * 5)
# Test frequencies in (-0.5, 0.5)
tests = np.concatenate((-tests[:-1][::-1], tests))
# Compute mean value of each with a window of nsamples
results = np.mean([np.exp(2j*np.pi*f*t2) for f in tests], axis=-1)

# lit skip
plt.plot(tests[len(tests)//2-50:len(tests)//2+50], np.abs(results[len(results)//2-50:len(results)//2+50]))
plt.vlines([-1.0/nsamples/2, 1.0/nsamples/2], 0, 0.5, color='tab:green')
plt.text(1.0/nsamples/2, 0.5, f'1/{nsamples*2}')
plt.vlines([-2.0/nsamples/2, 2.0/nsamples/2], 0, 0.5, color='tab:red')
plt.text(2.0/nsamples/2, 0.5, f'2/{nsamples*2}')
plt.vlines([-3.0/nsamples/2, 3.0/nsamples/2], 0, 0.5, color='tab:brown')
plt.text(3.0/nsamples/2, 0.5, f'3/{nsamples*2}')
plt.title(f'mean of {nsamples*2} samples')
plt.xlabel('frequency')
plt.ylabel('mag')
plt.savefig('mean-detector2-zoomed.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# The larger outputs in our tests are now associated with smaller frequencies, increasing this detector's ability to separate waves close to zero from other frequencies.
# There is a tradeoff here though.
# By increasing the frequency resolution of our detector, we have simultaneously decreased its time resolution: the first sample in our block is further separated from the last sample in time.
# In practice, we may want to detect a zero-frequency wave that is not always "on".
# When we detect it in a large block of samples, there is more ambiguity as to when it was active.
#
# All things considered, we have a very decent detector for near-zero-frequency waves in our signal.
# The magnitude of our signal's mean sample was close enough to zero that we can safely say that our waves have frequencies closer to one of the other labeled frequencies in this plot.
# Our detector returns `0` for these frequencies (waves that complete a whole number of cycles within the analyzed block of samples) so let's pick one of these frequencies to detect next and see if we can generalize our approach.
#
# Step 2
# ------
#
#
