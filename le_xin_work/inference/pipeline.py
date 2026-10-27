"""Chain v4 correction pipeline for ONE new singing recording: audio in, corrected notes (and a sheet) out.

    Basic Pitch (onset 0.99) -> 1 timing -> 2 split -> 3 delete -> 4 start snap (x2) -> 5 end snap -> 6 sheet

Steps 1-5 use the 7 final models in models/ (made by train_final_models.py, trained on training songs only).
Step 6 only draws the result: it estimates a tempo, writes a MIDI file and lets music21 round the notes to a
score (MusicXML). The rounding moves notes, so F1 is always measured on the step-5 note table, never on the sheet.

Usage:
    python pipeline.py song.wav --out out_folder           # needs basic-pitch + onnxruntime (see requirements.txt)
In code:
    from pipeline import Corrector
    result = Corrector().correct('song.wav')
    result['notes']          # note_id, recording_id, start, end, pitch (seconds, MIDI number)
"""
import argparse
import json
import string
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import soundfile as sf
import features as F

MODELS = Path(__file__).resolve().parent / 'models'
NOTE_COLUMNS = ['note_id', 'recording_id', 'start', 'end', 'pitch']
BP_CONFIG = dict(onset_threshold=0.99, frame_threshold=0.3, minimum_note_length=127.7, melodia_trick=True)  # tuning_001


# ------------------------------------------------------------------ step 0: Basic Pitch
def bp_model():
    """The bundled Basic Pitch ONNX model on CPU, set up as in notebooks/bp_tuning_core.make_model."""
    import onnxruntime as ort
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model
    path = build_icassp_2022_model_path(FilenameSuffix.onnx).resolve()
    model = Model(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    model.model = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    model.model_type = Model.MODEL_TYPES.ONNX
    return model


def bp_activations(audio_path, model):
    from basic_pitch.inference import run_inference
    out = run_inference(str(audio_path), model)
    return {k: np.asarray(out[k], dtype=np.float32) for k in ('onset', 'note', 'contour')}


def bp_notes(activations, recording_id):
    """Decode the activations with the project's chosen settings into the raw note table (same columns as
    basic_pitch_runs/correction_handover/val_features.csv)."""
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.note_creation import model_output_to_notes
    _, notes = model_output_to_notes(
        {k: np.array(activations[k], copy=True) for k in ('note', 'onset', 'contour')},
        onset_thresh=BP_CONFIG['onset_threshold'], frame_thresh=BP_CONFIG['frame_threshold'],
        min_note_len=int(np.round(BP_CONFIG['minimum_note_length'] / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))),
        melodia_trick=BP_CONFIG['melodia_trick'], min_freq=None, max_freq=None, infer_onsets=True,
        include_pitch_bends=True, multiple_pitch_bends=False, midi_tempo=120)
    pred = sorted([[float(n[0]), float(n[1]), int(n[2]), float(n[3])] for n in notes], key=lambda p: (p[0], p[1], p[2]))
    return raw_table(pred, recording_id)


def raw_table(pred, recording_id):
    """[start, end, pitch, amplitude] rows (sorted) -> raw note table (notebooks/build_correction_handover.py)."""
    rows = []
    for i, p in enumerate(pred):
        rows.append(dict(note_id=f'{recording_id}:n{i:05d}', recording_id=recording_id, start=p[0], end=p[1],
                         pitch=p[2], amplitude=p[3], duration=p[1] - p[0],
                         prev_gap=p[0] - pred[i - 1][1] if i else 0., next_gap=pred[i + 1][0] - p[1] if i + 1 < len(pred) else 0.))
    return pd.DataFrame(rows, columns=['note_id', 'recording_id', 'start', 'end', 'pitch', 'amplitude', 'duration',
                                       'prev_gap', 'next_gap'])


# ------------------------------------------------------------------ steps 1-5
class Corrector:
    def __init__(self, models=MODELS):
        self.s = json.loads((Path(models) / 'settings.json').read_text())
        self.m = {name: joblib.load(Path(models) / f'{name}.joblib')
                  for name in ['timing_start', 'timing_end', 'split', 'delete', 'snap_start', 'snap_end']}

    # step 1
    def timing(self, raw, recs):
        clues = pd.concat([F.timing_clues(recs[rid], g) for rid, g in raw.groupby('recording_id', sort=False)],
                          ignore_index=True)
        clues = raw[['note_id']].merge(clues, on='note_id', how='left', validate='one_to_one')
        X = pd.concat([raw[self.s['timing']['note_cols']].astype(float).reset_index(drop=True),
                       clues[self.s['timing']['clue_cols']]], axis=1)
        shift = np.column_stack([self.m['timing_start'].predict(X), self.m['timing_end'].predict(X)])
        gs, ge = self.s['timing']['gate_start'], self.s['timing']['gate_end']
        shift[np.abs(shift[:, 0]) < gs, 0] = 0
        shift[np.abs(shift[:, 1]) < ge, 1] = 0
        end_limit = raw.recording_id.map({rid: r.duration for rid, r in recs.items()}).to_numpy()
        out = raw[NOTE_COLUMNS].copy()
        out['start'] = np.maximum(0, raw['start'] + shift[:, 0])
        out['end'] = np.minimum(raw['end'] + shift[:, 1], end_limit)
        out['end'] = np.maximum(out['end'], out['start'] + 0.01)
        out['start'] = np.minimum(out['start'], out['end'] - 0.01)
        return out, clues

    # step 2
    def split(self, timed, recs, order):
        parts = [F.split_candidates(recs[rid], timed[timed.recording_id == rid]) for rid in order]
        parts = [p for p in parts if p is not None]
        if not parts:
            return timed[NOTE_COLUMNS].copy(), pd.DataFrame(columns=['note_id', 'recording_id', 'cut_time', 'prob'])
        c = F.split_note_clues(pd.concat(parts, ignore_index=True))
        c['prob'] = self.m['split'].predict_proba(c[self.s['split']['clues']].to_numpy(float))[:, 1]
        chosen = c[c.prob >= self.s['split']['threshold']]
        out, by_note = [], {nid: g.sort_values('t') for nid, g in chosen.groupby('note_id')}
        for n in timed.itertuples():
            cuts = by_note.get(n.note_id)
            if cuts is None or cuts.empty:
                out.append(dict(note_id=n.note_id, recording_id=n.recording_id, start=n.start, end=n.end, pitch=n.pitch))
                continue
            edges = [n.start, *cuts.t.tolist(), n.end]
            pitches = [cuts.piece_pitch_before.iloc[0], *cuts.piece_pitch_after.tolist()]
            assert len(edges) - 1 <= 26
            for k in range(len(edges) - 1):
                p = pitches[k]
                out.append(dict(note_id=f'{n.note_id}_{string.ascii_lowercase[k]}', recording_id=n.recording_id,
                                start=edges[k], end=edges[k + 1],
                                pitch=float(np.clip(round(p), 0, 127)) if not np.isnan(p) else n.pitch))
        acts = chosen.sort_values(['note_id', 't'])[['note_id', 'recording_id', 't', 'prob']].rename(columns={'t': 'cut_time'})
        return pd.DataFrame(out)[NOTE_COLUMNS], acts

    # step 3
    def delete(self, pool, raw, timing_clues, recs):
        f = F.delete_clues(pool, raw, timing_clues, recs)
        prob = self.m['delete'].predict_proba(np.nan_to_num(f[self.s['delete']['clues']].to_numpy(float), nan=-1))[:, 1]
        acts = pd.DataFrame(dict(note_id=f.note_id, recording_id=f.recording_id, prob=prob, deleted=prob >= self.s['delete']['threshold']))
        kept = pool[pool.note_id.isin(acts.note_id[~acts.deleted])].reset_index(drop=True)
        return kept, acts

    # step 4 (one round)
    def snap_start(self, notes, recs, margin):
        o = F.snap_options(notes, recs)
        s = pd.Series(self.m['snap_start'].predict(o[self.s['snap_start']['clues']]), index=np.arange(len(o)))
        g = s.groupby(o.note_id.to_numpy())
        e = np.exp(s - g.transform('max'))
        o['p'] = (e / e.groupby(o.note_id.to_numpy()).transform('sum')).to_numpy()   # softmax within each note
        new = _choose(o, margin)
        out = notes[NOTE_COLUMNS].copy()
        st = out.note_id.map(new)
        out['start'] = np.where(st.notna(), st, out.start)
        out['start'] = np.minimum(out.start, out.end - 0.01)
        return out, o

    # step 5
    def snap_end(self, notes, recs, order):
        o = F.end_options(notes, recs, order)
        o['p'] = self.m['snap_end'].predict_proba(o[self.s['snap_end']['clues']])[:, 1]
        best = _choose(o, self.s['snap_end']['margin'])
        out = notes[NOTE_COLUMNS].copy()
        e = out.note_id.map(best)
        out['end'] = np.maximum(np.where(e.notna(), e, out.end), out.start + 0.01)
        return out, o

    def run(self, raw, recs, order=None):
        """raw: Basic Pitch note table (one or more recordings); recs: recording_id -> features.Recording.
        Returns every stage's note table and what each step changed."""
        order = list(dict.fromkeys(raw.recording_id)) if order is None else order
        t1, tclues = self.timing(raw, recs)
        t2, cuts = self.split(t1, recs, order)
        t3, dele = self.delete(t2, raw, tclues, recs)
        t4a, _ = self.snap_start(t3, recs, self.s['snap_start']['margin_round1'])
        t4, _ = self.snap_start(t4a, recs, self.s['snap_start']['margin_round2'])
        t5, _ = self.snap_end(t4, recs, order)
        return dict(basic_pitch=raw[NOTE_COLUMNS], timing=t1, split=t2, delete=t3, start_snap=t4, notes=t5,
                    cuts=cuts, deletions=dele)

    def correct(self, audio_path, recording_id=None, bp=None):
        """One audio file -> corrected notes. Runs Basic Pitch unless bp=(activations, raw notes) is given."""
        audio_path = Path(audio_path)
        rid = recording_id or audio_path.stem
        if bp is None:
            acts = bp_activations(audio_path, bp_model())
            raw = bp_notes(acts, rid)
        else:
            acts, raw = bp
        rec = F.Recording(rid, audio_path, acts, sf.info(str(audio_path)).duration)
        if raw.empty:
            empty = raw[NOTE_COLUMNS]
            return dict(basic_pitch=empty, notes=empty)
        return self.run(raw, {rid: rec})


class Avg:
    """Split model: mean probability of a random forest and a gradient boosting model (split_v2/s5_models.Avg)."""
    def __init__(self, models):
        self.ms = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


def _choose(o, margin):
    """Move to the best non-stay option if its score beats 'stay' by >= margin. Returns note_id -> new time."""
    stay_p = o[o.is_stay == 1].set_index('note_id').p
    moves = o[o.is_stay == 0].dropna(subset=['p'])
    best = moves.loc[moves.groupby('note_id').p.idxmax()].set_index('note_id')
    return best.loc[best.p - stay_p.reindex(best.index) >= margin, 't']


# ------------------------------------------------------------------ step 6: sheet (display only, never scored)
def estimate_tempo(audio_path):
    import librosa
    y, sr = librosa.load(str(audio_path), sr=F.SR, mono=True)
    bpm = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr))[0])
    return float(round(bpm)) if 30 <= bpm <= 300 else 120.0


def write_midi(notes, path, bpm, program=53):
    """Corrected notes -> MIDI at the estimated tempo (program 53 = 'Voice Oohs'). Times stay in seconds."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=program, name='Voice')
    for n in notes.sort_values('start').itertuples():
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(n.pitch), start=float(n.start), end=float(n.end)))
    pm.instruments.append(inst)
    pm.write(str(path))


def write_sheet(notes, audio_path, out_dir, title):
    """Tempo -> MIDI -> music21 (rounds notes to 16ths/triplets) -> MusicXML. Overlaps are trimmed for the sheet
    only (a sung melody is one line), so the score is easier to read."""
    import music21
    out_dir = Path(out_dir)
    sheet = notes.sort_values('start').reset_index(drop=True).copy()
    nxt = sheet.start.shift(-1)
    sheet['end'] = np.where(nxt.notna() & (sheet.end > nxt), np.maximum(nxt, sheet.start + 0.01), sheet.end)
    bpm = estimate_tempo(audio_path)
    mid, xml = out_dir / f'{title}.mid', out_dir / f'{title}.musicxml'
    write_midi(sheet, mid, bpm)
    score = music21.converter.parse(str(mid), quantizePost=True, quarterLengthDivisors=(4, 3))
    score.metadata = music21.metadata.Metadata(title=title)
    score.write('musicxml', fp=str(xml))
    return dict(bpm=bpm, midi=str(mid), musicxml=str(xml))


def main():
    ap = argparse.ArgumentParser(description='Correct Basic Pitch notes for one singing recording and draw a sheet.')
    ap.add_argument('audio')
    ap.add_argument('--out', default='.')
    ap.add_argument('--id', default=None, help='recording id (default: file name)')
    ap.add_argument('--no-sheet', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rid = a.id or Path(a.audio).stem
    r = Corrector().correct(a.audio, rid)
    r['basic_pitch'].to_csv(out / f'{rid}_basic_pitch.csv', index=False)
    r['notes'].to_csv(out / f'{rid}_corrected.csv', index=False)
    print(f"{len(r['basic_pitch'])} Basic Pitch notes -> {len(r['notes'])} corrected notes: {out / f'{rid}_corrected.csv'}")
    if not a.no_sheet and len(r['notes']):
        s = write_sheet(r['notes'], a.audio, out, rid)
        print(f"tempo {s['bpm']:.0f} BPM; MIDI {s['midi']}; sheet {s['musicxml']}")


if __name__ == '__main__':
    main()
