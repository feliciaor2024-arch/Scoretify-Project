"""Audio for the app's Play button: the corrected notes as soft tones, alone and mixed with the singer.

Browsers cannot play MIDI, so the notes are rendered to WAV here. Both files share the recording's timeline,
so the score cursor and the singer stay in step.
"""
import librosa
import numpy as np
import soundfile as sf

SR = 22050
HARMONICS = ((1, 1.0), (2, 0.35), (3, 0.15))    # a little brighter than a sine, still easy to follow
ATTACK, RELEASE = 0.01, 0.04                     # seconds; avoids clicks at note edges


def tones(notes, length, sr=SR):
    out = np.zeros(length)
    for n in notes:
        a, b = int(n['start'] * sr), min(int(n['end'] * sr), length)
        if b <= a:
            continue
        t = np.arange(b - a) / sr
        f = 440.0 * 2 ** ((n['pitch'] - 69) / 12)
        wave = sum(w * np.sin(2 * np.pi * f * k * t) for k, w in HARMONICS)
        envelope = np.clip(np.minimum(t / ATTACK, ((b - a) / sr - t) / RELEASE), 0, 1)
        out[a:b] += wave * envelope
    return out


def peak(x, level):
    m = np.abs(x).max()
    return x * (level / m) if m > 0 else x


def render(notes, audio_path, synth_path, mix_path):
    """Write the tones-only and tones + voice WAVs. Returns their length in seconds."""
    voice, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    length = max(len(voice), int((max(n['end'] for n in notes) + 0.1) * SR))
    notes_audio = peak(tones(notes, length), 0.6)
    voice = peak(np.pad(voice, (0, length - len(voice))), 0.7)
    sf.write(str(synth_path), notes_audio, SR, subtype='PCM_16')
    sf.write(str(mix_path), peak(voice + 0.6 * notes_audio, 0.9), SR, subtype='PCM_16')
    return length / SR
