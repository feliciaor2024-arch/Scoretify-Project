"""Snap v3 experiments: shared pieces (Le Xin's snap-to-moment start picker, training songs only).

Re-implements the Step 1/2/4 functions of notebooks/correction_handover/le_xin_snap_model.ipynb so options can be
rebuilt with a wider window, extra moment kinds, or around new starts (repeat snap). Checked to reproduce the
notebook's option table (snap_candidates_train_v2.csv) and its snap v2 scores before any new experiment.

Rules kept: clues come from the audio, Basic Pitch activations and the predicted notes only; the answer key
(references_train.csv) is used only for labels and scoring; data/test is never opened; no TextGrid.
"""
from pathlib import Path
import os, pickle, sys, importlib.util
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'notebooks' / 'correction_handover'
OUT = Path(__file__).resolve().parent
BASE_CACHE = OUT / 'cache'
CACHE = OUT / os.environ.get('SNAP_CACHE', 'cache')        # other input chains keep their own option/score caches
CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import correction_utils as cu
import librosa

DATA, EXPORTS, LX_CACHE = ROOT / 'basic_pitch_runs' / 'correction_handover', HERE / 'exports', HERE / 'le_xin_cache'
BPACT = ROOT / 'le_xin_work' / 'bp_activations' / 'train'
read = lambda path: pd.read_csv(path, dtype={'note_id': str, 'recording_id': str, 'sample_id': str})

manifest = pd.read_csv(DATA / 'training_recording_manifest.csv', dtype={'recording_id': str, 'sample_id': str})
assert manifest['split'].eq('train').all()                          # test stays locked
references = read(DATA / 'references_train.csv')                     # answer key: labels and scoring only
folds = pd.read_csv(DATA / 'folds.csv').set_index('sample_id')['fold']
chain = read(Path(os.environ['SNAP_CHAIN']) if 'SNAP_CHAIN' in os.environ else EXPORTS / 'combined_final_train_oof_v1.csv')
rec_fold = manifest.set_index('recording_id').sample_id.map(folds)
ALL = sorted(folds[manifest.sample_id.unique()].unique())

SR, HOP = 22050, 256
FRAME = HOP / SR
FAR = 12.0
F0 = np.load(HERE / 'hann_qian_pitch_model' / 'f0_tracks.npz')

CLUES = ['dt', 'abs_dt', 'is_stay', 'is_attack', 'is_louder', 'is_voice_on', 'is_pitch_jump', 'attack', 'louder',
         'pitch_jump', 'voiced_before', 'voiced_after', 'gap_before', 'gap_after', 'note_dur', 'note_pitch',
         'is_piece', 'prev_gap', 'n_options', 'attack_rank']
BP_CLUES = ['bp_onset_at', 'bp_onset_any', 'bp_note_before', 'bp_note_after', 'bp_note_rise']
SETTLE_CLUES = ['settle_dt', 'settle_abs', 'has_settle', 'match_100', 'steady_100', 'settle_rank', 'match_rank']
NEIGH_CLUES = ['to_prev_start', 'to_prev_end', 'to_next_start']
COMPARED = ['attack', 'louder', 'pitch_jump', 'bp_onset_at', 'bp_note_rise']
COMP_CLUES = [f'{a}_{b}' for a in COMPARED for b in ['vs_stay', 'is_max']]
CLUES_V2 = CLUES + BP_CLUES + SETTLE_CLUES + NEIGH_CLUES + COMP_CLUES
EXTRA_KIND_CLUES = ['is_bp_onset', 'is_settle']                      # only in tables built with extra_kinds=True


# ---------------------------------------------------------------- audio curves (cached once per recording)
def _curves(rec):
    y, _ = librosa.load(ROOT / rec.audio_path, sr=SR, mono=True)
    flux = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    flux = flux / (np.percentile(flux, 99) + 1e-9)
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=HOP)[0], ref=np.max)
    rise = np.diff(db, prepend=db[0])
    f0 = F0[f'{rec.recording_id}_f0']
    n = min(len(flux), len(rise), len(f0))
    midi = librosa.hz_to_midi(f0[:n])
    voiced = ~np.isnan(midi)
    jump = np.abs(np.diff(np.nan_to_num(midi), prepend=0.0))
    jump[~voiced] = 0; jump[1:][~voiced[:-1]] = 0
    return dict(t=librosa.frames_to_time(np.arange(n), sr=SR, hop_length=HOP),
                flux=flux[:n], rise=rise[:n], midi=midi, voiced=voiced, jump=jump)

_CURVES = None
def curves():
    global _CURVES
    if _CURVES is None:
        f = BASE_CACHE / 'curves_train.pkl'
        if f.exists():
            _CURVES = pickle.loads(f.read_bytes())
        else:
            _CURVES = {}
            for rec in manifest.itertuples():
                assert rec.audio_path.startswith('data/train/')
                _CURVES[rec.recording_id] = _curves(rec)
            f.write_bytes(pickle.dumps(_CURVES))
    return _CURVES

def peaks(v, height):
    return np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] >= height)) + 1

def moment_frames(c):
    return {'attack': peaks(c['flux'], 0.1), 'louder': peaks(c['rise'], 1.0),
            'voice_on': np.flatnonzero(c['voiced'][1:] & ~c['voiced'][:-1]) + 1,
            'pitch_jump': np.flatnonzero(c['jump'] >= 0.5)}

def sung_gap(c, a, b, pitch):
    seg = c['midi'][(c['t'] >= a) & (c['t'] < b)]
    seg = seg[~np.isnan(seg)]
    return min(abs(np.median(seg) - pitch), FAR) if len(seg) else FAR

def voiced_share(c, a, b):
    m = (c['t'] >= a) & (c['t'] < b)
    return float(c['voiced'][m].mean()) if m.any() else 0.0


# ---------------------------------------------------------------- Basic Pitch activations (cached by snap_bp_clues.py)
MIDI_OFFSET, W, SIDE = 21, 2, 5
def bp_times(n_frames):
    """basic_pitch.note_creation.model_frames_to_time, copied (constants: 22050 Hz, hop 256, 2 s windows)."""
    fps = SR // HOP
    annot_n = fps * 2
    n_samples = SR * 2 - HOP
    original = librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP)
    window_offset = (HOP / SR) * (annot_n - (n_samples / HOP)) + 0.0018
    return original - window_offset * np.floor(np.arange(n_frames) / annot_n)

_ACT = {}
def bp_act(rid):
    if rid not in _ACT:
        arr = np.load(BPACT / f'{rid}.npz')
        _ACT[rid] = (arr['onset'], arr['note'], bp_times(len(arr['onset'])))
    return _ACT[rid]

def bp_onset_peaks(rid, pitch, height=0.3):
    """Frames (on the 256-hop audio grid) where Basic Pitch's onset signal at the note's pitch (+-1) peaks."""
    onset, _, times = bp_act(rid)
    b = int(round(pitch)) - MIDI_OFFSET
    v = onset[:, max(b - 1, 0):min(b + 2, onset.shape[1])].max(axis=1)
    return times[peaks(v, height)]


# ---------------------------------------------------------------- options + clues
def build_options(notes, win=0.10, extra_kinds=False):
    """Options (stay + real moments within +-win s of each start) with all v2 clues and labels.
    notes: complete note table (note_id, recording_id, start, end, pitch). With win=0.10 and extra_kinds=False this
    reproduces le_xin_cache/snap_candidates_train_v2.csv."""
    C = curves()
    notes = notes.sort_values(['recording_id', 'start']).copy()
    notes['prev_gap'] = (notes.start - notes.groupby('recording_id').end.shift(1)).fillna(0.0)
    rows = []
    for rid, nts in notes.groupby('recording_id', sort=False):
        c = C[rid]
        kinds = moment_frames(c)
        all_frames = np.unique(np.concatenate(list(kinds.values())))
        near_kind = {k: set(np.concatenate([f - 1, f, f + 1])) for k, f in kinds.items()}
        for n in nts.itertuples():
            here = min(int(round(n.start / FRAME)), len(c['t']) - 1)
            cand = all_frames
            extra = {}
            if extra_kinds:
                bpf = np.round(bp_onset_peaks(rid, n.pitch) / FRAME).astype(int)
                st = settle_time(c, n.start, n.pitch, win)
                sf = np.array([int(round(st / FRAME))]) if not np.isnan(st) else np.array([], int)
                bpf, sf = bpf[(bpf >= 0) & (bpf < len(c['t']))], sf[(sf >= 0) & (sf < len(c['t']))]
                extra = {'bp_onset': set(np.concatenate([bpf - 1, bpf, bpf + 1])),
                         'settle': set(np.concatenate([sf - 1, sf, sf + 1]))}
                cand = np.unique(np.concatenate([all_frames, bpf, sf]))
            frames = cand[(np.abs(c['t'][cand] - n.start) <= win) & (np.abs(cand - here) > 2)]
            merged = [f for i, f in enumerate(frames) if i == 0 or f - frames[i - 1] > 2]
            options = [(n.start, here, True)] + [(c['t'][f], f, False) for f in merged]
            for t, i, stay in options:
                lo, hi = max(i - 1, 0), min(i + 2, len(c['t']))
                near = set(range(i - 2, i + 3)) if stay else {i}
                r = dict(note_id=n.note_id, recording_id=rid, t=t, dt=t - n.start, is_stay=float(stay),
                         **{f'is_{k}': float(bool(near & near_kind[k])) for k in kinds})
                if extra_kinds:
                    r.update({f'is_{k}': float(bool(near & s)) for k, s in extra.items()})
                r.update(attack=c['flux'][lo:hi].max(), louder=c['rise'][lo:hi].max(), pitch_jump=c['jump'][lo:hi].max(),
                         voiced_before=voiced_share(c, t - 0.05, t), voiced_after=voiced_share(c, t, t + 0.05),
                         gap_before=sung_gap(c, t - 0.05, t, n.pitch), gap_after=sung_gap(c, t, t + 0.05, n.pitch),
                         note_dur=n.end - n.start, note_pitch=n.pitch, is_piece=float('~' in n.note_id),
                         prev_gap=n.prev_gap)
                rows.append(r)
    cands = pd.DataFrame(rows)
    g = cands.groupby('note_id', sort=False)
    cands['n_options'] = g.t.transform('size')
    cands['attack_rank'] = g.attack.rank(ascending=False, method='min')
    cands['abs_dt'] = cands.dt.abs()
    cands = add_labels(cands, notes)
    cands = add_bp_clues(cands)
    cands = add_new_clues(cands, notes)
    cands['sample_id'] = cands.recording_id.map(manifest.set_index('recording_id').sample_id)
    cands['fold'] = cands.sample_id.map(folds)
    return cands

def settle_time(c, s0, p, win=0.15):
    """First time within +-win of s0 where the sung pitch stays within 0.5 st of p for 4 frames (NaN if none)."""
    on = np.abs(c['midi'] - p) < 0.5
    for j in np.flatnonzero((c['t'] >= s0 - win) & (c['t'] <= s0 + win)):
        if on[j:j + 4].size == 4 and on[j:j + 4].all():
            return c['t'][j]
    return np.nan

REF = {rid: g for rid, g in references.groupby('recording_id')}
def true_start(rid, start, pitch):
    g = REF.get(rid)
    if g is None:
        return np.nan
    g = g[(np.abs(g.pitch - pitch) < 0.5) & (np.abs(g.start - start) <= 0.15)]
    return g.start.iloc[np.argmin(np.abs(g.start - start))] if len(g) else np.nan

def targets(notes):
    """note_id -> true start (answer key; labels/scoring only). Always measured from the ORIGINAL chain start so
    every version is judged on the same 4,285 labelled notes."""
    return pd.Series({n.note_id: true_start(n.recording_id, n.start, n.pitch) for n in notes.itertuples()})

_TARGET = None
def chain_targets():
    global _TARGET
    if _TARGET is None:
        _TARGET = targets(chain)
    return _TARGET

def add_labels(cands, notes):
    target = targets(notes)
    out = cands.copy()
    out['_target'] = out.note_id.map(target)
    out['label'] = ((out.t - out._target).abs() <= 0.05).astype(int)
    out['labelled_note'] = out._target.notna()
    return out.drop(columns='_target')

def add_bp_clues(c):
    """Same numbers as le_xin_work/tools/snap_bp_clues.py, from its cached activations."""
    vals = np.zeros((len(c), 4))
    for rid, idx in c.groupby('recording_id', sort=False).indices.items():
        onset, note, times = bp_act(rid)
        for r in idx:
            t, p = c.t.iat[r], c.note_pitch.iat[r]
            i = int(np.argmin(np.abs(times - t)))
            b = int(round(p)) - MIDI_OFFSET
            nb = list(range(max(b - 1, 0), min(b + 2, onset.shape[1])))
            lo, hi = max(i - W, 0), min(i + W + 1, len(onset))
            vals[r] = [float(onset[lo:hi, nb].max()), float(onset[lo:hi].max()),
                       float(note[max(i - SIDE, 0):i, nb].max(axis=1).mean()) if i > 0 else 0.0,
                       float(note[i:i + SIDE, nb].max(axis=1).mean())]
    for j, col in enumerate(BP_CLUES[:4]):
        c[col] = vals[:, j]
    return c

def add_new_clues(options, notes):
    c = options.copy()
    c['bp_note_rise'] = c.bp_note_after - c.bp_note_before
    n = notes.sort_values(['recording_id', 'start']).copy()
    g = n.groupby('recording_id')
    n['prev_start'], n['prev_end'], n['next_start'] = g.start.shift(1), g.end.shift(1), g.start.shift(-1)
    n = n.set_index('note_id')
    c['to_prev_start'] = (c.t - c.note_id.map(n.prev_start)).fillna(5.0).clip(upper=5.0)
    c['to_prev_end'] = (c.t - c.note_id.map(n.prev_end)).fillna(5.0).clip(-5.0, 5.0)
    c['to_next_start'] = (c.note_id.map(n.next_start) - c.t).fillna(5.0).clip(upper=5.0)
    settle_dt = np.full(len(c), np.nan); match_100 = np.full(len(c), FAR); steady_100 = np.full(len(c), FAR)
    for rid, idx in c.groupby('recording_id', sort=False).indices.items():
        tt, midi = F0[f'{rid}_t'], librosa.hz_to_midi(F0[f'{rid}_f0'])
        settled = {}
        for i in idx:
            nid, t, p = c.note_id.iat[i], c.t.iat[i], c.note_pitch.iat[i]
            if nid not in settled:
                s0 = n.at[nid, 'start']
                on = np.abs(midi - p) < 0.5
                hit = [j for j in np.flatnonzero((tt >= s0 - 0.15) & (tt <= s0 + 0.15)) if on[j:j + 4].size == 4 and on[j:j + 4].all()]
                settled[nid] = tt[hit[0]] if hit else np.nan
            settle_dt[i] = t - settled[nid]
            seg = midi[(tt >= t) & (tt < t + 0.10)]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                match_100[i] = min(abs(np.median(seg) - p), FAR)
            if len(seg) > 1:
                steady_100[i] = min(np.std(seg), FAR)
    settle_dt = pd.Series(settle_dt, c.index)
    c['has_settle'] = settle_dt.notna().astype(float)
    c['settle_abs'] = settle_dt.abs().fillna(1.0)
    c['settle_dt'] = settle_dt.fillna(1.0)
    c['match_100'], c['steady_100'] = match_100, steady_100
    g = c.groupby('note_id', sort=False)
    stay = c[c.is_stay == 1].set_index('note_id')
    for col in COMPARED:
        c[f'{col}_vs_stay'] = c[col] - c.note_id.map(stay[col])
        c[f'{col}_is_max'] = (c[col] == g[col].transform('max')).astype(float)
    c['match_rank'] = g.match_100.rank(method='min')
    c['settle_rank'] = g.settle_abs.rank(method='min')
    return c


# ---------------------------------------------------------------- scoring (same scorer as cu.evaluate_end_to_end, memoised)
_spec = importlib.util.spec_from_file_location('_bp_core', ROOT / 'notebooks' / 'bp_tuning_core.py')
core = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(core)
_REF_ARR = {rid: g[['start', 'end', 'pitch']].to_numpy() for rid, g in references.groupby('recording_id')}
_MEMO = {}
EMPTY = np.zeros((0, 3))

def evaluate(table, rec_ids=None):
    """Same numbers as cu.evaluate_end_to_end on the chosen recordings (all training recordings by default)."""
    man = manifest if rec_ids is None else manifest[manifest.recording_id.isin(rec_ids)]
    est = {rid: g[['start', 'end', 'pitch']].to_numpy() for rid, g in table.groupby('recording_id')}
    rows = []
    for record in man.to_dict('records'):
        rid = record['recording_id']
        e = est.get(rid, EMPTY)
        key = (rid, e.tobytes())
        if key not in _MEMO:
            _MEMO[key] = core.score_notes(_REF_ARR.get(rid, EMPTY), e)
        rows.append({**record, **_MEMO[key]})
    return core.aggregate(rows)[1]

def apply_starts(new_start, table=None):
    out = (chain if table is None else table)[cu.NOTE_COLUMNS].copy()
    s = out.note_id.map(new_start)
    out['start'] = np.where(s.notna(), s, out.start)
    out['start'] = np.minimum(out.start, out.end - 0.01)
    return out

def recs_of(fold_set):
    return set(manifest.recording_id[manifest.recording_id.map(rec_fold).isin(fold_set)])

def f1_on(fold_set, new_start, table=None):
    s = evaluate(apply_starts(new_start, table), recs_of(fold_set))
    return s['song_macro_f1'], s['song_macro_offset_f1']

def labelled_right(table):
    """Share of the 4,285 labelled notes (true start from the original chain) now starting within 50 ms of it."""
    tg = chain_targets().dropna()
    st = table.set_index('note_id').start.reindex(tg.index)
    return float(((st - tg).abs() <= 0.05 + 1e-9).mean())


# ---------------------------------------------------------------- choosing + nested margin
MARGINS = [-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 9.0]

def choose(prob, margin, table):
    c = table[['note_id', 't', 'is_stay']].assign(p=prob)
    stay_p = c[c.is_stay == 1].set_index('note_id').p
    moves = c[c.is_stay == 0].dropna(subset=['p'])
    best = moves.loc[moves.groupby('note_id').p.idxmax()].set_index('note_id')
    go = best.p - stay_p.reindex(best.index) >= margin
    return best.loc[go, 't']

def oof(fitpred, table, fold_set):
    """Scores for rows in fold_set, each from a model trained on the other folds of fold_set (labelled notes only)."""
    p = np.full(len(table), np.nan)
    lab = table.labelled_note.to_numpy()
    for k in fold_set:
        test = (table.fold == k).to_numpy()
        train = table.fold.isin([f for f in fold_set if f != k]).to_numpy() & lab
        p[test] = fitpred(table[train], table[test])
    return p

def nested(fitpred, table, name, margins=MARGINS, chooser=choose, base=None):
    """Out-of-fold scores + margin chosen per held-out fold on the other 4 folds (inner out-of-fold).
    Scores are cached in cache/<name>.npz. base: the note table the options were built from (default chain)."""
    f = CACHE / f'{name}.npz'
    if f.exists():
        z = np.load(f)
        outer, inner = z['outer'], {k: z[f'inner{k}'] for k in ALL}
    else:
        outer = oof(fitpred, table, ALL)
        inner = {k: oof(fitpred, table, [j for j in ALL if j != k]) for k in ALL}
        np.savez(f, outer=outer, **{f'inner{k}': v for k, v in inner.items()})
    picked, starts = {}, []
    note_fold = table.drop_duplicates('note_id').set_index('note_id').fold
    for k in ALL:
        others = [j for j in ALL if j != k]
        m = max(margins, key=lambda g: f1_on(others, chooser(inner[k], g, table), base)[0])
        picked[int(k)] = m
        ch = chooser(outer, m, table)
        starts.append(ch[ch.index.map(note_fold) == k])
    return outer, inner, pd.concat(starts), picked

def report(name, starts, margins=None, base=None):
    t = apply_starts(starts, base)
    s = evaluate(t)
    assert s['recordings'] == len(manifest)
    return dict(version=name, starts_moved=int((t.start.to_numpy() != (chain if base is None else base).start.to_numpy()).sum()),
                labelled_right=round(100 * labelled_right(t), 1),
                onset_F1=round(s['song_macro_f1'], 4), offset_F1=round(s['song_macro_offset_f1'], 4),
                margins=margins)


def load_v2():
    """The notebook's option table with v2 clues (same rows/order as snap_candidates_train_v1.csv)."""
    c = read(LX_CACHE / 'snap_candidates_train_v2.csv')
    c['sample_id'] = c.recording_id.map(manifest.set_index('recording_id').sample_id)
    c['fold'] = c.sample_id.map(folds)
    return c
