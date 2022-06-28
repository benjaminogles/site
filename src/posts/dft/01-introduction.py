
# lit text
#
# The Discrete Fourier Transform From Scratch
# ===========================================
#
# In this post I show how to derive the formula for the one-dimensional, complex discrete Fourier transform (DFT).
#
# Motivating Example
# ------------------
#
# Radio waves can be used to implement wireless communication systems by modulating the phase or magnitude of a transmitted wave with message data that can be recovered by a complementary process at the receiving end of the system.
# Complex numbers are a natural fit for representing such signals in digital form because they also consist of a phase and magnitude.
# As such, many receivers implement a digitization process that amounts to sampling the phase and magnitude of received signals and recording this information as a sequence of complex numbers in computer memory for further analysis.
# Suppose we are given such a recording of wireless morse code transmissions and tasked to extract and decode each message.
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
# Inspecting the data shows that the morse code messages are contained in the signal's magnitude.
# When only a single transmitter is active in the data, plotting the magnitude of each sample yields a graphic that can be immediately decoded.
#
# lit skip

plt.plot(np.abs(morse_code_ook_signal("HI MOM", .25 * sample_rate, sample_rate)))
plt.title("One Transmitter Active")
plt.ylabel("Magnitude")
plt.xlabel("Sample")
plt.savefig("morse-code-1-user-time.png")
plt.close()

# lit execute
# lit unskip
# lit text
#
# The _dits_ and _dahs_ of morse code are represented by sequences of samples with non-zero magnitude while the _spaces_ are represented by sequences of samples with zero magnitude.
# This is a digital modulation scheme known as On-Off-Keying (OOK).
# A sinusoidal signal with constant frequency and constant magnitude was transmitted and the transmitter was simply turned on and off with each `1` and `0` in the morse code data.
# The transmitted signal is called the carrier and its frequency is called the carrier frequency.
#
# It is much more difficult to read messages from a magnitude plot when multiple carriers are being transmitted.
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
plt.ylabel("Magnitude")
plt.xlabel("Sample")
plt.savefig("morse-code-3-users-time.png")
plt.close()

# lit execute
# lit unskip
# lit text
#
# The magnitude of each sample in this case has been affected by three different transmitters.
# Can we separate the effects of each transmitter to isolate each message?
# The answer is yes!
# If each transmitter used a different carrier frequency, then we can digitally filter these samples to wipe out the effects of all but one carrier.
# But in order to do that, we need to actually know the values of the three different carrier frequencies.
#
# The DFT can help us achieve that goal by transforming this sequence of _time domain_ samples into the _frequency domain_ where each sample will then represent a phase and magnitude weight of a particular frequency component in the signal, rather than a sampling instant.
# It can also be used to implement the digital filter, but that is out of scope for the purpose of this example.
# The minimum and maximum frequencies represented in the DFT output are directly determined by the sampling rate used to digitize the time domain data.
# In the plot below, I've normalized the frequency axis so that these frequencies are assigned `-0.5` and `0.5` respectively.
# We can inspect this plot to see which carrier frequencies are being transmitted in this sequency of samples.
#
# lit skip

plt.plot(np.fft.fftshift(np.fft.fftfreq(nfft)), np.abs(np.fft.fftshift(np.fft.fft(sig, n=nfft))))
plt.title("DFT of Samples Plotted Above")
plt.ylabel("Magnitude")
plt.xlabel("Normalized Frequency")
plt.savefig("morse-code-3-users-freq.png")
plt.close()

# lit execute
