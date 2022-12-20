
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
# It may be helpful to read these short posts first for some basic background information:
#
# - [Complex Signals](/posts/complex-signals/), to understand complex numbers and their relationship to sinusoidal signals
# - [Nyquist Frequency](/posts/nyquist-frequency/), to understand the basics of sampling
#
# See (/posts/dft/) for a high level overview of the DFT and links to other resources for understanding it.
#
# Example Problem
# ---------------
#
# We will generate several arrays of complex samples, each one representing a pure sinusoid (constant frequency and magnitude).
# Our task will be to estimate the individual frequencies of these waves by analyzing the sum of their samples.
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
# Random phase offset for each wave
offsets = rng.uniform(0, 2 * np.pi, nwaves)
# Instantaneous phase of each wave
phi = [2 * np.pi * freqs[i] * t + offsets[i] for i in range(nwaves)]
# Complex samples of each wave
waves = [np.exp(1j * p) for p in phi]
# Combine
signal = np.sum(waves, axis=0)

# lit skip
assert nsamples == 128, 'text may reference this number'
plt.plot(np.real(signal), label='real')
plt.plot(np.imag(signal), label='imag')
plt.legend()
plt.title('signal')
plt.xlabel('sample')
plt.ylabel('amplitude')
plt.savefig('signal.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# Note that we could scale `t` and `freqs` by a sample rate to express the problem in physical units (e.g seconds and Hertz).
# But to keep things more general, I have chosen to use a normalized sample rate of `1`.
#
# Step 1
# ------
#
# We will start by solving a simplified version of this problem: determining whether our sum includes a wave with a frequency near zero.
# We can do this by examining the sum's mean value.
#
# Sinusoids always oscillate around a mean value of zero.
# Recall that the arithmetic mean is a linear operator.
#

# Two random signals
a = rng.uniform(size=nsamples)
b = rng.uniform(size=nsamples)
# Mean of sum is equal to sum of means
assert np.isclose(np.mean(a+b), np.mean(a)+np.mean(b))

# lit execute
# lit text
#
# So the mean value of a sum of sinusoids is also theoretically zero.
# But in practice we can only ever analyze a finite window of any signal.
# The mean value of a sinusoid within a finite window will only be zero if that window includes a whole number of cycles.
# Otherwise, the unfinished cycle at the end of the window will bias its mean value (and the mean value of any sum it belongs to) away from zero.
#
# Inversely proportional to `nsamples/`
#

# lit skip
assert np.abs(np.mean(signal)) < 0.2, "did the rng seed change?"
# lit unskip

print(f'mean={np.mean(signal)}')

# lit text
#
# The mean value of our signal is only slightly biased away from zero in both the real and imaginary parts.
#
# lit execute
#
# However,
#
# A wave with a frequency near zero will hardly change within
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
c = rng.uniform()
assert np.isclose(c * np.mean(signal), c * np.sum(np.mean(waves, axis=-1)))
# Time-Invariant (delaying the input just delays the same output)
# Our np.mean system is trivially time invariant
# It does not even consider sample index in its computation
# Here is an example of a time-invariant system
system = lambda t, x: t * x
# assert not np.allclose(system(t, b), system(t-1, b))

# lit execute
# lit text
#
# 
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
