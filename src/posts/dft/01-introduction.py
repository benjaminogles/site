
# lit text
#
# The Discrete Fourier Transform From Scratch
# ===========================================
#
# In this post I use a simple example problem to motivate the derivation of the one-dimensional, complex discrete Fourier transform (DFT).
# At the end of the post, I extend this example problem to show how it may appear in practice.
#
# Example Problem
# ---------------
#
# I'm going to create a test signal by adding samples from two sinusoids oscillating at different frequencies.
# I will pick the frequencies at random and the problem is solved when we find a way to estimate these frequencies values by analyzing the test signal.
#

import numpy as np
import matplotlib.pyplot as plt

# Use a sample rate of 1 so that sample units are synonymous with whatever
# physical units we would normally use e.g. time
# In other words, our frequencies have units cycles per sample
nsamples = 128

# I want at least a full cycle's worth of samples
min_freq = 1/nsamples
# I want several samples per cycle for smoother plots
max_freq = 1/(nsamples/8)

# Pick two frequencies
rng = np.random.default_rng(seed=192837465)
freqs = rng.uniform(min_freq, max_freq, size=2)

# Generate nsamples of a sine wave at each frequency
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
# If we were allowed to analyze each sine wave individually, we could estimate their frequencies by counting cycles.
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
# So we need a function, or filter, that copies its input in a frequency-dependent manner.
# As a start, it should pass lower frequency oscillations while rejecting higher frequency oscillations.
# In fact, let's start by trying to only accept oscillations from the lowest possible frequency.
#
# A frequency of 0 corresponds to a value that never changes.
# So a filter targeting this frequency will reject all oscillations.
# And copy through only whatever constant offset existed in the input.
# A reasonable implementation then might compute mean values along a sliding window of samples.
# This would smooth oscillations down to whatever constant level they are oscillating around.

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

plt.plot(np.abs(morse_code_ook_signal("HI MOM", .25 * sample_rate, sample_rate)))
plt.title("One Transmitter Active")
plt.ylabel("Magnitude")
plt.xlabel("Sample")
plt.savefig("morse-code-1-user-time.png")
plt.close()

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
