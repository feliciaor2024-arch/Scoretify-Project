"""Step 3: rebuild the old candidates from the cached tracks (check they match the notebook cache), study the
true cut points the old finder misses, and add extra candidate sources (timbre change, loudness valley,
Basic Pitch onset peak, smaller pitch change). Answer key used ONLY for labels / recall numbers."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import common as C
import clues as K

manifest, references, folds, raw, timed = C.load()


def frame_tracks(tr, bp):
    """Per-frame extra signals: timbre change, loudness valley depth, Basic Pitch onset (any pitch) on our frame grid,
    small pitch change (same 5-frame medians as the old finder)."""
    midi, db, mfcc = tr['midi'], tr['db'], tr['mfcc'][1:]
    n = len(midi)
    cs = np.concatenate([np.zeros((mfcc.shape[0], 1)), np.cumsum(mfcc, axis=1)], axis=1)
    spec = np.zeros(n)
    for f in range(5, n - 5):
        mb, ma = (cs[:, f] - cs[:, f - 5]) / 5, (cs[:, f + 5] - cs[:, f]) / 5
        spec[f] = 1 - mb @ ma / (np.linalg.norm(mb) * np.linalg.norm(ma) + 1e-9)
    valley = np.zeros(n)
    for f in range(9, n - 9):
        valley[f] = min(db[f - 9:f - 2].max(), db[f + 2:f + 9].max()) - db[f - 3:f + 4].min()
    on = bp['onset'].max(axis=1)
    bt = K.bp_times(len(on))
    bpon = np.interp(np.arange(n) * C.FRAME, bt, on)
    return dict(spec=spec, valley=valley, bpon=bpon)


def peaks(score, frames, floor):
    s = score[frames]
    return {int(f) for k, f in enumerate(frames) if s[k] >= floor and s[k] == s[max(0, k - 3):k + 4].max()}


def run(rid, sample_id):
    tr = C.load_tracks(rid)
    bp = dict(np.load(C.BPACT / f'{rid}.npz'))
    ft = frame_tracks(tr, bp)
    notes = timed[timed.recording_id == rid]
    refs = references[references.recording_id == rid]
    args = (tr['midi'], tr['voiced'], tr['vprob'], tr['flux'], tr['db'])
    old = pd.DataFrame([r for n in notes.itertuples() for r in C.candidates_for_note(n, *args)])

    def extra(n, lo, hi):
        frames = np.arange(lo, hi + 1)
        before = np.array([C.med(tr['midi'], i - C.SIDE, i) for i in frames])
        after = np.array([C.med(tr['midi'], i, i + C.SIDE) for i in frames])
        change = np.nan_to_num(np.abs(after - before), nan=0.0)
        out = {}
        for f in {int(frames[k]) for k in range(len(frames))
                  if change[k] >= 0.3 and change[k] == change[max(0, k - 3):k + 4].max()}:
            out[f] = 'pitch_small'
        for f in peaks(ft['bpon'], frames, 0.3):
            out.setdefault(f, 'bp_onset')
        for f in peaks(ft['valley'], frames, 3.0):
            out.setdefault(f, 'valley')
        for f in peaks(ft['spec'], frames, 0.15):
            out.setdefault(f, 'timbre')
        return out
    new = pd.DataFrame([r for n in notes.itertuples() for r in C.candidates_for_note(n, *args, extra_frames=extra)])
    truth = C.true_cut_times(notes, refs)
    # signals at every needed true cut (for the recall study)
    miss = []
    for nid, times in truth.items():
        for b in times:
            f = int(round(b / C.FRAME))
            w = slice(max(f - 4, 0), f + 5)
            oc = old[old.note_id == nid] if len(old) else old
            nc = new[new.note_id == nid] if len(new) else new
            n = notes.set_index('note_id').loc[nid]
            miss.append(dict(note_id=nid, t=b, old_hit=bool(len(oc)) and np.abs(oc.t - b).min() <= 0.05,
                             new_hit=bool(len(nc)) and np.abs(nc.t - b).min() <= 0.05,
                             near_edge=min(b - n.start, n.end - b) < C.EDGE,
                             spec=ft['spec'][w].max(), valley=ft['valley'][w].max(), bpon=ft['bpon'][w].max(),
                             flux=tr['flux'][w].max()))
    return old, new, truth, pd.DataFrame(miss)


if __name__ == '__main__':
    res = Parallel(n_jobs=4, verbose=1)(delayed(run)(r.recording_id, r.sample_id) for r in manifest.itertuples())
    old = pd.concat([r[0] for r in res], ignore_index=True)
    new = pd.concat([r[1] for r in res], ignore_index=True)
    truth = {k: v for r in res for k, v in r[2].items()}
    study = pd.concat([r[3] for r in res], ignore_index=True)
    ref_cache = pd.read_csv(C.HERE / 'le_xin_cache' / 'split_candidates_train_v1.csv',
                            dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
    same = len(old) == len(ref_cache) and (old.note_id.to_numpy() == ref_cache.note_id.to_numpy()).all() \
        and np.allclose(old.t, ref_cache.t)
    print('Rebuilt old candidates match the notebook cache:', same, len(old), len(ref_cache))
    new = C.label(new, truth)
    needed = sum(map(len, truth.values()))
    print('needed true cuts', needed)
    print(study[['old_hit', 'new_hit', 'near_edge']].mean().round(3).to_dict())
    print('missed by old, signal medians (missed vs found):')
    print(study.groupby('old_hit')[['spec', 'valley', 'bpon', 'flux']].median().round(3))
    for s in ['old', 'pitch_small', 'bp_onset', 'valley', 'timbre']:
        sub = new[new.source == s]
        print(f'{s:12s} candidates {len(sub):6d}  real {int(sub.label.sum()):5d}  share real {sub.label.mean():.3f}')
    new.to_csv(C.CACHE / 'cands_v2_train.csv', index=False)
    study.to_csv(C.CACHE / 'recall_study_train.csv', index=False)
