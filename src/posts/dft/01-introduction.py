
# lit text
#
# The Discrete Fourier Transform From Scratch
# ===========================================
#
# In this post I show how to derive the one-dimensional, complex discrete Fourier transform (DFT) from first principles.
#
# Motivating Example
# ------------------
#
# We are given a data file, procured by intercepting and digitizing wireless morse code transmissions.
# The data file contains complex samples, meaning we can compute the amplitude and phase of the intercepted signal at each sample.
# Our task is to extract and decode each morse code message.
#
# As background, we learn that morse code is made up of symbols called _dits_, _dahs_ and _spaces_.
# These symbols are differentiated by duration.
# A _dit_ and _space_ have the same duration with the _space_ defined as the absence of a transmitted _dit_.
# The _dah_ is equal to three _dits_.
# A sequence of _dits_ and _dahs_, separated by one _space_ each, encodes a letter.
# Each letter in a word is separated by three _spaces_ and each word in a message is separated by seven _spaces_.
#
# lit skip

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=19281730)

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

# lit unskip
# lit text
#
# There are sections of the data file where it is clear that only one transmitter was active.
# If I plot the amplitude of each sample in such a section, we can clearly see that it carries the morse code message from the active transmitter.
#
# lit skip

plt.plot(np.abs(morse_code_ook_signal("HI MOM", .25 * sample_rate, sample_rate)))
plt.title("One Transmitter Active")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.savefig("morse-code-1-user-time.png")
plt.close()

# lit execute
# lit unskip
# lit text
#
# The _dits_ and _dahs_ of morse code are represented by sequences of samples with amplitude `1` while the _spaces_ are represented by sequences of samples with amplitude `0`.
# This is a digital modulation scheme known as On-Off-Keying (OOK).
# A constant frequency signal with amplitude `1` was transmitted and the transmitter was simply turned on and off with each `1` and `0`.
# The signal that was transmitted is called the carrier and its frequency is called the carrier frequency.
#
# These samples could be decoded without any further preprocessing (it should read "HI MOM" by the way, if I've done it correctly).
# But it is more difficult to read messages from the data file's amplitude plot when multiple transmitters are active.
#
# lit skip

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
plt.plot(np.abs(sig))
plt.title("Three Transmitters Active")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.savefig("morse-code-3-users-time.png")
plt.close()

# lit execute
# lit unskip
# lit text
#
# The amplitude of these samples have been affected by three different transmitters.
# Can we separate the effects of each transmitter to decode the three messages?
# The answer is yes!
# If each transmitter uses a different carrier frequency, then we can digitally filter these samples to wipe out the effects of all but one carrier.
# But before we can do that, we need to know which carrier frequencies exist.
#
# This is where the DFT comes in.
# We need to determine which carrier frequencies are active in the section of the data file containing these samples.
# We can also use it to implement the digital filter but that is out of scope for this example.
# Computing the DFT of this sequence of samples in time yields a sequence of samples in frequency i.e. the DFT outputs are equally spaced by some amount of Hz just like the original samples are equally spaced by some amount of seconds.
# The frequency axis is centered around the tune frequency used by the radio to collect the data and I've arbitrarily placed that frequency at `0` while scaling the relative min and max to plus or minus `0.5`.
#
# lit skip

plt.plot(np.fft.fftshift(np.fft.fftfreq(nfft)), np.abs(np.fft.fftshift(np.fft.fft(sig, n=nfft))))
plt.title("DFT of Samples Plotted Above")
plt.ylabel("Amplitude")
plt.xlabel("Normalized Frequency")
plt.savefig("morse-code-3-users-freq.png")
plt.close()

# lit execute
# lit unskip
# lit text
#
# Looking at the peaks in the DFT, we can see the three active carrier frequencies are close to `-0.25`, `0` and `0.25`.
