
# lit text
#
# The Discrete Fourier Transform From Scratch
# ===========================================
#
# In this post I use a simple example problem to motivate the derivation of the one-dimensional, complex discrete Fourier transform (DFT).
#
# Example Problem
# ---------------
#
# Estimate the frequencies of two discrete sinusoidal signals after they are added together.
#

import numpy as np
import matplotlib.pyplot as plt

# Pick an arbitrary number of samples to generate
nsamples = 128

# Require each wave to complete at least one cycle in nsamples
min_freq = 1/nsamples
# Require at least nsamples/8 samples per cycle for smoother plotting
max_freq = 1/(nsamples/8)

# Pick two frequencies
rng = np.random.default_rng(seed=192837465)
freqs = rng.uniform(min_freq, max_freq, size=2)

# Generate nsamples of a sine wave at each frequency and add
sine1 = np.sin(2 * np.pi * freqs[0] * np.arange(nsamples))
sine2 = np.sin(2 * np.pi * freqs[1] * np.arange(nsamples))
signal = sine1 + sine2

# lit skip
plt.plot(sine1, label='sine1')
plt.plot(sine2, label='sine2')
plt.plot(signal, label='signal')
plt.legend()
plt.savefig('sines.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# If the signal contained a singl.
#

def count_zero_crossings(x, threshold=0.25):
    # Assign True/False based on sign after thresholding to larger values
    positive = x[np.abs(x) > abs(threshold)] > 0
    # Compare adjacent values and count mismatches
    return np.count_nonzero(positive[:-1] != positive[1:])

# Two zero crossings per cycle and we estimate the number of cycles per sample
estimates = [count_zero_crossings(x)/2/nsamples for x in (sine1, sine2)]

print(f'Actual Frequencies: {freqs}')
print(f'Estimated Frequencies: {estimates}')

# lit text
#
# This gives us reasonably close estimates without much effort.
#
# lit execute
# lit text
#
# For this example specifically, you could probably adapt this method to work on the test signal as well because the two frequencies are easy enough to pick out visually.
# But it would be nice to identify a method that also works on signals containing more than two component frequencies.

print(count_zero_crossings(signal)/2/nsamples)

# lit text
#
# As it is, this cycle counting method is only useful for estimating the lower frequency.
#
# lit execute
# lit text
#
# Filtering
# ---------
#
# We aren't allowed to analyze the sinusoids individually, but that idea leads us in the right direction of trying to separate them from each other somehow.
# At the very least, if we can find a way to separate "low" frequency sinusoids from "high" frequency sinusoids in a signal, then we will immediately be able to use the cycle counting method on each "half" of our test signal in this example.
#
# So we need a function, or filter, that copies its input to the output in a frequency-dependent manner.
# As a start, it should accept or amplify lower frequency oscillations and reject or attenuate higher frequency oscillations.
# In fact, let's start by trying to only accept oscillations from the lowest possible frequency.
#
# A frequency of 0 corresponds to a constant valued signal.
# So a filter targeting this frequency should reject all oscillations, leaving only the mean value.
# This hints at an implementation that computes the mean of samples in a sliding window.
#

def moving_average(x, size, mode='valid'):
    return np.convolve(x, np.ones(size)/size, mode=mode)

# lit text
#
# The convolve function generalizes the idea of a moving average by allowing you to specify a weight for each position in the sliding window.
# So a moving average reduces to specifying an equal weight for each position, scaled by the size of the window.
# The mode parameter controls how the edges of the signal are handled and we use "valid" mode by default to only compute output samples when the window and signal completely overlap.
#
# To effectively filter down to a signal's mean value, the window size must be large enough to incorporate peaks and valleys of the lowest frequency signals we want to reject.
#

# Estimate the number of samples in one cycle of the lower frequency sine wave
samples_per_cycle = 1/min(estimates)
# Try a window size that won't filter the lower frequency as much
small_window_result = moving_average(signal, int(samples_per_cycle/2), 'same')
# Try a window size that will filter the lower frequency more
large_window_result = moving_average(signal, int(samples_per_cycle*2), 'same')
# Use "same" mode to zero-pad edges and get an output sample for every input sample

# lit skip
plt.plot(signal, label='signal')
plt.plot(small_window_result, label='small-window')
plt.plot(large_window_result, label='large-window')
plt.legend()
plt.savefig('averaged-signal.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# If we take this reasoning to the limit, it becomes clear that the moving average filter will never isolate the zero-frequency component completely.
# We will always have a finite number of input samples and the signal can always contain very low frequency oscillations that won't complete a full cycle within that finite window.
# Let's run some tests to see exactly which frequencies are effectively filtered by a moving average filter.
#

# Test a bunch of frequencies in [0, 0.5)
test_freqs = np.linspace(0, 0.5, 1000, endpoint=False)
# Make each test signal as long as the filter
t = np.arange(nsamples)
# Using "valid" mode, we will only get one output sample per test
output = [ moving_average(np.exp(2j*np.pi*f*t), nsamples) for f in test_freqs ]

# lit skip
plt.plot(test_freqs, np.abs(output))
plt.savefig('moving-average-mag-response.png')
plt.close()

# lit unskip
# lit execute
# lit text
#
# As expected, the filter output is very close to the mean value when the input is a high frequency sine wave.
# Let's zoom in on the lower end.
#

# lit skip
plt.plot(test_freqs[:nsamples//4], np.abs(output)[:nsamples//4])
plt.savefig('moving-average-mag-response-zoomed.png')
plt.close()

# lit unskip
# lit execute
#
# lit skip
#
# Radio waves can be used to implement wireless communication systems by modulating a _carrier wave_'s phase and/or magnitude with message data that is recovered by a complementary demodulating process at the receiving end of the system.
# Complex numbers are a natural fit for representing communication signals in digital form because they are defined in terms of phase and magnitude.
# As such, many receivers implement a digitization process that amounts to sampling the phase and magnitude of an analog input and recording this information as a sequence of complex numbers in computer memory.
# Suppose we are given a recording of complex samples collected by a receiver that was used to intercept wireless morse code transmissions.
# We are tasked to extract and decode each morse code message.
#

sample_rate = 1.0
morse_code_symbol_period = 1/sample_rate * 40
morse_code_symbol_map = {
        'A': [1,0,1,1,1],
        'B': [1,1,1,0,1,0,1,0,1],
        'C': [1,1,1,0,1,0,1,1,1,0,1],
        'D': [1,1,1,0,1,0,1],
        'E': [1],
        'F': [1,0,1,0,1,1,1,0,1],
        'G': [1,1,1,0,1,1,1,0,1],
        'H': [1,0,1,0,1,0,1],
        'I': [1,0,1],
        'J': [1,0,1,1,1,0,1,1,1,0,1,1,1],
        'K': [1,1,1,0,1,0,1,1,1],
        'L': [1,0,1,1,1,0,1,0,1],
        'M': [1,1,1,0,1,1,1],
        'N': [1,1,1,0,1],
        'O': [1,1,1,0,1,1,1,0,1,1,1],
        'P': [1,0,1,1,1,0,1,1,1,0,1],
        'Q': [1,1,1,0,1,1,1,0,1,0,1,1,1],
        'R': [1,0,1,1,1,0,1],
        'S': [1,0,1,0,1],
        'T': [1,1,1],
        'U': [1,0,1,0,1,1,1],
        'V': [1,0,1,0,1,0,1,1,1],
        'W': [1,0,1,1,1,0,1,1,1],
        'X': [1,1,1,0,1,0,1,0,1,1,1],
        'Y': [1,1,1,0,1,0,1,1,1,0,1,1,1],
        'Z': [1,1,1,0,1,1,1,0,1,0,1],
        '1': [1,0,1,1,1,0,1,1,1,0,1,1,1],
        '2': [1,0,1,0,1,1,1,0,1,1,1,0,1,1,1],
        '3': [1,0,1,0,1,0,1,1,1,0,1,1,1,0],
        '4': [1,0,1,0,1,0,1,1,1,0,1,1,1,0],
        '5': [1,0,1,0,1,0,1,0,1,0],
        '6': [1,1,1,0,1,0,1,0,1,0,1],
        '7': [1,1,1,0,1,1,1,0,1,0,1,0,1],
        '8': [1,1,1,0,1,1,1,0,1,1,1,0,1,0,1],
        '9': [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1],
        '0': [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],
        ' ': [0]
        }

def morse_code_symbols(message):
    mapped = [morse_code_symbol_map[u] + [0,0,0] for u in message.upper()]
    return [sym for symlist in mapped for sym in symlist]

def morse_code_ook_signal(message, fc, fs, phi=0):
    syms = morse_code_symbols(message)
    dur = len(syms) * morse_code_symbol_period
    t = np.linspace(0, dur, int(dur * fs), endpoint=False)
    carrier = np.exp(1j * (2 * np.pi * fc * t + phi))
    return carrier * [syms[int(samp_inst/morse_code_symbol_period)] for samp_inst in t]

def random_morse_code_ook_signal(nletters, fc, fs, phi=0):
    keys = list(morse_code_symbol_map.keys())
    idx = rng.integers(len(keys), size=nletters)
    return morse_code_ook_signal(''.join(map(str, [keys[i] for i in idx])), fc, fs, phi)

def next_power_of_2(n):
    return 1 if n == 0 else 2 ** (n-1).bit_length()

#
# At the beginning of the recording, only one morse code transmitter is active.
# Plotting each sample's magnitude yields a graphic that can be immediately decoded.
#

# plt.plot(np.abs(morse_code_ook_signal("HI MOM", .25 * sample_rate, sample_rate)))
# plt.title("One Transmitter Active")
# plt.ylabel("Magnitude")
# plt.xlabel("Sample")
# plt.savefig("morse-code-1-user-time.png")
# plt.close()

#
# The _dits_ and _dahs_ of morse code are represented by sequences of samples with non-zero magnitude while the _spaces_ are represented by sequences of samples with zero magnitude.
# This is a digital modulation scheme known as On-Off-Keying (OOK).
#
# It is much more difficult to decode a magnitude plot of samples later in the recording when multiple morse code transmitters are active.
#

message_len = 8
max_possible_symbols = message_len * max([len(m) for m in morse_code_symbol_map.values()])
max_possible_dur = max_possible_symbols * morse_code_symbol_period
max_possible_signal_len = int(max_possible_dur * sample_rate)
nfft = next_power_of_2(max_possible_signal_len)
sigs = [
        random_morse_code_ook_signal(message_len, round(-0.25 * nfft)/nfft * sample_rate, sample_rate),
        random_morse_code_ook_signal(message_len, round(0.0   * nfft)/nfft * sample_rate, sample_rate),
        random_morse_code_ook_signal(message_len, round(0.25  * nfft)/nfft * sample_rate, sample_rate),
        ]
max_sig_len = max(len(s) for s in sigs)
sig = np.sum([np.pad(s, (0, max_sig_len-len(s))) for s in sigs], axis=0)
# plt.plot(np.abs(sig))
# plt.title("Three Transmitters Active")
# plt.ylabel("Magnitude")
# plt.xlabel("Sample")
# plt.savefig("morse-code-3-users-time.png")
# plt.close()

#
# In this case, the magnitude of each sample has been affected by three different carrier waves.
# Can we separate out the effects of each carrier to isolate and decode their messages?
#
