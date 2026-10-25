"""Step 7: extra timbre / consonant clues for the v2 candidates (audio only):
timbre change over ~100 ms windows, strongest timbre change nearby, high-frequency (>4 kHz) energy share burst
(consonants like s/sh/ch/t), zero-crossing-rate peak. Appends columns to clues_v2_train.csv -> clues_v2t_train.csv."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import librosa
import common as C

manifest, references, folds, raw, timed = C.load()
cands = pd.read_csv(C.CACHE / 'clues_v2_train.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
paths = manifest.set_index('recording_id').audio_path

def run(rid, g):
    y, _ = librosa.load(C.ROOT / paths[rid], sr=C.SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=C.HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=C.SR, n_fft=1024)
    hf = S[freqs > 4000].sum(0) / (S.sum(0) + 1e-12)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=C.HOP)[0]
    mf = C.load_tracks(rid)['mfcc'][1:]
    n = min(len(hf), len(zcr), mf.shape[1])
    cs = np.concatenate([np.zeros((mf.shape[0], 1)), np.cumsum(mf[:, :n], axis=1)], axis=1)
    def spec(f, w):
        a, b = max(f - w, 0), min(f + w, n)
        if f - a < 2 or b - f < 2:
            return 0.0
        mb, ma = (cs[:, f] - cs[:, a]) / (f - a), (cs[:, b] - cs[:, f]) / (b - f)
        return 1 - mb @ ma / (np.linalg.norm(mb) * np.linalg.norm(ma) + 1e-9)
    rows = []
    for c in g.itertuples():
        f = min(max(int(round(c.t / C.FRAME)), 1), n - 1)
        w = slice(max(f - 4, 0), f + 3)
        rows.append(dict(spec9=spec(f, 9), spec_near=max(spec(j, 5) for j in range(max(f - 3, 1), min(f + 4, n))),
                         hf_peak=hf[w].max(), hf_rise=hf[w].max() - np.median(hf[max(f - 12, 0):f + 12]),
                         zcr_peak=zcr[w].max(), zcr_rise=zcr[w].max() - np.median(zcr[max(f - 12, 0):f + 12])))
    return pd.DataFrame(rows, index=g.index)

if __name__ == '__main__':
    parts = Parallel(n_jobs=4, verbose=1)(delayed(run)(rid, g) for rid, g in cands.groupby('recording_id'))
    out = pd.concat([cands, pd.concat(parts).loc[cands.index]], axis=1)
    out.to_csv(C.CACHE / 'clues_v2t_train.csv', index=False)
    from sklearn.metrics import roc_auc_score
    for col in ['spec_change', 'spec9', 'spec_near', 'hf_peak', 'hf_rise', 'zcr_peak', 'zcr_rise']:
        print(col, round(roc_auc_score(out.label, out[col]), 3))
