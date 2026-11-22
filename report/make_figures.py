"""Draw every figure in the final report from the files in this repository.

    python report/make_figures.py            # all figures
    python report/make_figures.py 4 6        # only figs 4 and 6

Writes report/figures/fig<N>_<name>.png. Install its packages first: pip install -r report/requirements.txt
Reads only saved files: the sample manifests, the GTSinger annotations and audio in data/, and the one test run in
final_pipeline/test_run/. Nothing is retrained or re-run; the test numbers are re-scored with the same scorer as
final_pipeline/inference/score_test.py.

Figure      Source
Fig. 1      selected_100.csv (language x split, vocal_range)
Fig. 2      one training recording (GTS021, English, alto, Breathy_Group, clip 0002): audio + annotation
Fig. 3      diagram of the final pipeline (no data)
Fig. 4      final_pipeline/test_run/test_scores.csv
Fig. 5      one test recording (GTS018, English, tenor, Control_Group of Breathy, clip 0005): test note tables
Fig. 6      final_pipeline/test_run/test_scores_by_song.csv
"""
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'report' / 'figures'
TEST = ROOT / 'final_pipeline' / 'test_run'

BLUE, LIGHT, GOLD, RED, GREY = '#2a6f97', '#7b8fd1', '#c9a227', '#d1495b', '#9aa5ad'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200})

FIG2_RECORDING = '5fe72323ae3e5a6c18dd'   # GTS021, English, alto, Breathy_Group, clip 0002 (training)
FIG5_RECORDING = 'a17354f2bfe3c15d97c0'   # GTS018, English, tenor, Breathy technique, Control_Group, clip 0005 (test)


def load_score_test():
    """final_pipeline/inference/score_test.py as a module: same reference parser and scorer as the test run."""
    spec = importlib.util.spec_from_file_location('score_test', ROOT / 'final_pipeline/inference/score_test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def annotated_notes(json_path):
    """Reference notes as scored: aligned note_start / note_end, pitch-0 events (rests, breaths) removed."""
    rows = []
    for item in json.loads((ROOT / json_path).read_text(encoding='utf-8')):
        for p, s, e in zip(item.get('note', []), item.get('note_start', []), item.get('note_end', [])):
            if p:
                rows.append((float(s), float(e), int(p)))
    return pd.DataFrame(rows, columns=['start', 'end', 'pitch'])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path.relative_to(ROOT))


def fig1():
    songs = pd.read_csv(ROOT / 'selected_100.csv')
    split_names = [('train', 'Training', BLUE), ('val', 'Validation', LIGHT), ('test', 'Test', GOLD)]
    counts = songs.groupby(['language', 'split']).size().unstack(fill_value=0)
    counts = counts.loc[sorted(counts.index, reverse=True)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={'width_ratios': [1.4, 1]})
    left = np.zeros(len(counts))
    for key, label, colour in split_names:
        a.barh(counts.index, counts[key], left=left, color=colour, label=label)
        left += counts[key].to_numpy()
    for y, total in enumerate(left):
        a.text(total + 0.15, y, f'{int(total)}', va='center', fontsize=8)
    a.set_xlabel('Songs')
    a.set_title('(a) Songs per language and split')
    a.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.08), frameon=False)
    voices = songs.vocal_range.value_counts().reindex(['Soprano', 'Alto', 'Tenor', 'Bass'])
    bars = b.bar(voices.index, voices.values, color=BLUE, width=0.6)
    b.bar_label(bars, fontsize=8)
    b.set_ylabel('Songs')
    b.set_title('(b) Songs per voice type')
    save(fig, 'fig1_dataset.png')


def fig2():
    import librosa
    import librosa.display
    rec = pd.read_csv(ROOT / 'basic_pitch_runs/correction_handover/recording_manifest.csv').set_index('recording_id')
    row = rec.loc[FIG2_RECORDING]
    notes = annotated_notes(row.json_path)
    sr, hop = 22050, 256                                          # as in final_pipeline/inference/features.py
    y, _ = librosa.load(ROOT / row.audio_path, sr=sr, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1100, sr=sr, hop_length=hop)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    # Hide pYIN's guesses in near-silence (frames over 35 dB below the loudest), where it tracks background noise.
    level = librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=hop)[0], ref=np.max)[:len(f0)]
    midi = np.where(voiced & (level > -35), librosa.hz_to_midi(f0), np.nan)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    for n in notes.itertuples():
        a.plot([n.start, n.end], [n.pitch, n.pitch], color=GOLD, lw=6, solid_capstyle='butt')
    a.plot([], [], color=GOLD, lw=6, label='Annotated notes (answer key)')
    a.plot(times, midi, color=BLUE, lw=1.2, label='Sung pitch (pYIN)')
    a.set_ylabel('Pitch (MIDI)')
    a.set_ylim(48, 70)
    a.legend(loc='lower center', ncol=2, frameon=False, fontsize=8)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop)), ref=np.max)
    librosa.display.specshow(spec, sr=sr, hop_length=hop, x_axis='time', y_axis='log', ax=b, cmap='magma')
    b.set_ylim(64, 4096)
    b.set_ylabel('Frequency (Hz)')
    b.set_xlabel('Time (s)')
    a.set_xlim(0, len(y) / sr)
    save(fig, 'fig2_pitch_vs_notes.png')


def fig3():
    boxes = [  # (column, row, title, subtitle, is a trained step)
        (0, 0, 'Sung audio', '', False), (1, 0, '0. Basic Pitch', 'pretrained, tuned decoder', False),
        (2, 0, '1. Timing', '2 random forests', True), (3, 0, '2. Split', 'RF + gradient boosting', True),
        (3, 1, '3. Delete', 'gradient boosting', True), (2, 1, '4. Start snap', 'LightGBM ranker, 2 rounds', True),
        (1, 1, '5. End snap', 'random forest', True), (0, 1, '6. Sheet music', 'tempo, 32nd grid, music21', False)]
    w, h, gx, gy = 2.2, 1.0, 0.5, 0.7
    fig, ax = plt.subplots(figsize=(9, 2.8))
    centre = {}
    for col, row, title, sub, trained in boxes:
        x, y = col * (w + gx), -row * (h + gy)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', lw=1,
                                    fc=BLUE if trained else '#eef1f3', ec=BLUE if trained else GREY))
        colour = 'white' if trained else 'black'
        ax.text(x + w / 2, y + h * 0.62, title, ha='center', va='center', weight='bold', color=colour)
        ax.text(x + w / 2, y + h * 0.3, sub, ha='center', va='center', fontsize=7, color=colour)
        centre[title] = (x, y)
    order = [b[2] for b in boxes]
    for first, second in zip(order, order[1:]):
        (x1, y1), (x2, y2) = centre[first], centre[second]
        if y1 == y2:   # same row: right edge -> left edge, or the reverse on the second row
            start, end = ((x1 + w, y1 + h / 2), (x2, y2 + h / 2)) if x2 > x1 else ((x1, y1 + h / 2), (x2 + w, y2 + h / 2))
        else:          # down from Split to Delete
            start, end = (x1 + w / 2, y1), (x2 + w / 2, y2 + h)
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.set_xlim(-0.2, 4 * (w + gx))
    ax.set_ylim(-(h + gy) - 0.2, h + 0.2)
    ax.axis('off')
    save(fig, 'fig3_pipeline.png')


def fig4():
    scores = pd.read_csv(TEST / 'test_scores.csv').set_index('stage')
    labels = {'basic_pitch': 'Basic Pitch\nalone', 'timing': '1. Timing', 'split': '2. Split', 'delete': '3. Delete',
              'start_snap': '4. Start\nsnap', 'notes': '5. End snap\n(final)', 'sheet_notes': '6. Sheet,\nread back'}
    scores = scores.loc[list(labels)]
    x = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for offset, column, colour, label in [(-0.2, 'onset_f1', BLUE, 'Onset + pitch F1'),
                                          (0.2, 'offset_f1', GOLD, 'Offset F1')]:
        bars = ax.bar(x + offset, scores[column], width=0.38, color=colour, label=label)
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in scores[column]], fontsize=7, padding=2)
    ax.axvline(len(scores) - 1.5, color=GREY, ls=':', lw=1)
    ax.set_xticks(x, [labels[s] for s in scores.index], fontsize=8)
    ax.set_ylabel('F1 (song-macro)')
    ax.legend(frameon=False, loc='upper left')
    save(fig, 'fig4_test_by_step.png')


def fig5():
    from librosa import midi_to_hz
    from mir_eval.transcription import match_notes
    st = load_score_test()
    man = st.read(TEST / 'test_manifest.csv')
    refs = st.references(man.query('recording_id == @FIG5_RECORDING'))
    ref = refs[['start', 'end', 'pitch']].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, sharey=True)
    for ax, (table, name) in zip(axes, [('test_basic_pitch.csv', '(a) Basic Pitch alone'),
                                        ('test_notes.csv', '(b) After the five correction steps')]):
        est = st.read(TEST / table).query('recording_id == @FIG5_RECORDING')[['start', 'end', 'pitch']]
        est = est.to_numpy(dtype=float)
        pairs = match_notes(ref[:, :2], midi_to_hz(ref[:, 2]), est[:, :2], midi_to_hz(est[:, 2]),
                            onset_tolerance=.05, pitch_tolerance=50., offset_ratio=None, strict=False)
        hit = {e for _, e in pairs}                     # same matching as bp_tuning_core.score_notes (onset F1)
        f1 = 2 * len(pairs) / (len(ref) + len(est))
        for s, e, p in ref:
            ax.add_patch(Rectangle((s, p - 0.5), e - s, 1, color=GOLD, alpha=0.45, lw=0))
        for i, (s, e, p) in enumerate(est):
            ax.plot([s, e], [p, p], color=BLUE if i in hit else RED, lw=3, solid_capstyle='butt')
            ax.plot([s, s], [p - 0.45, p + 0.45], color=BLUE if i in hit else RED, lw=1.5)
        ax.set_title(f'{name}: {len(pairs)} of {len(ref)} real notes matched, onset F1 {f1:.2f}', loc='left',
                     fontsize=9)
        ax.set_ylabel('Pitch (MIDI)')
    axes[0].add_patch(Rectangle((0, 0), 0, 0, color=GOLD, alpha=0.45, label='Real note (answer key)'))
    axes[0].plot([], [], color=BLUE, lw=3, label='Predicted note, counted correct')
    axes[0].plot([], [], color=RED, lw=3, label='Predicted note, counted wrong')
    axes[0].legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=8)
    axes[1].set_xlabel('Time (s)')
    axes[1].autoscale_view()
    save(fig, 'fig5_example_recording.png')


def fig6():
    songs = pd.read_csv(TEST / 'test_scores_by_song.csv').sort_values('notes')
    y = np.arange(len(songs))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hlines(y, songs.basic_pitch, songs.notes, color=GREY, lw=1.2)
    ax.scatter(songs.basic_pitch, y, color=GREY, s=18, zorder=3, label='Basic Pitch alone')
    ax.scatter(songs.notes, y, color=BLUE, s=18, zorder=3, label='Final corrected notes')
    ax.set_yticks(y, songs.sample_id, fontsize=7)
    ax.set_xlabel('Onset + pitch F1')
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'fig6_test_by_song.png')


FIGURES = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}

if __name__ == '__main__':
    wanted = [int(a) for a in sys.argv[1:]] or list(FIGURES)
    for number in wanted:
        FIGURES[number]()
