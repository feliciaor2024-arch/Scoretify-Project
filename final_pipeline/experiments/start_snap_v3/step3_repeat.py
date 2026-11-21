"""Step 3: repeat snap. Run the picker a second time on its own output (options rebuilt around the new starts).
Fair version: for held-out fold k, round 2 uses the round-1 model that never saw fold k, and the round-2 margin
is chosen on the other 4 folds using their inner out-of-fold round-1 output and inner models only."""
import sys, time
import numpy as np, pandas as pd
from common import *
from pickers import *

def repeat(fp, table, name, win=0.10, extra=False, clues=None):
    outer, inner, starts1, m1 = nested(fp, table, name)
    base1 = apply_starts(starts1)
    lab = table.labelled_note.to_numpy()
    def fit_pred(train_folds, te):
        tr = table[table.fold.isin(train_folds).to_numpy() & lab]
        return fp(tr, te)
    f = CACHE / f'{name}_round2.pkl'
    if f.exists():
        r2 = pd.read_pickle(f)
    else:
        r2 = {}
        opts = build_options(base1, win, extra)                    # round-2 options around the round-1 output
        p = np.full(len(opts), np.nan)
        for k in ALL:
            te = (opts.fold == k).to_numpy()
            p[te] = fit_pred([j for j in ALL if j != k], opts[te])
        r2['outer'] = (opts, p)
        for k in ALL:                                              # inner: what the other 4 folds would have seen
            others = [j for j in ALL if j != k]
            s1 = choose(inner[k], m1[int(k)], table)
            b1 = apply_starts(s1[s1.index.map(table.drop_duplicates('note_id').set_index('note_id').fold).isin(others)])
            o = build_options(b1, win, extra)
            q = np.full(len(o), np.nan)
            for j in others:
                te = (o.fold == j).to_numpy()
                q[te] = fit_pred([i for i in others if i != j], o[te])
            r2[k] = (b1, o, q)
        pd.to_pickle(r2, f)
    opts, p = r2['outer']
    picked, starts2 = {}, []
    note_fold = opts.drop_duplicates('note_id').set_index('note_id').fold
    for k in ALL:
        others = [j for j in ALL if j != k]
        b1, o, q = r2[k]
        m = max(MARGINS, key=lambda g: f1_on(others, choose(q, g, o), b1)[0])
        picked[int(k)] = m
        ch = choose(p, m, opts)
        starts2.append(ch[ch.index.map(note_fold) == k])
    starts2 = pd.concat(starts2)
    final = apply_starts(starts2, base1)
    return base1, final, m1, picked, starts2

def row(title, table, margins):
    s = evaluate(table)
    return dict(version=title, starts_moved=int((table.start.to_numpy() != chain.start.to_numpy()).sum()),
                labelled_right=round(100 * labelled_right(table), 1), onset_F1=round(s['song_macro_f1'], 4),
                offset_F1=round(s['song_macro_offset_f1'], 4), margins=margins)

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'rf'
    if which == 'rf':
        tab, fp, name, win, extra = load_v2(), fp_rf(), 'rf_v2', 0.10, False
    else:                                   # e.g. "hgb_w150x": picker + saved option table from step 2
        p, tag = which.split('_', 1)
        win, extra = int(tag[1:4]) / 1000, tag.endswith('x')
        tab = pd.read_pickle(CACHE / f'options_{tag}.pkl')
        clues = CLUES_V2 + (EXTRA_KIND_CLUES if extra else [])
        fp = {'rf': fp_rf, 'hgb': fp_hgb, 'rfhgb': fp_rf_hgb, 'lgbrank': lambda c: fp_lgb_rank('lambdarank', c)}[p](clues)
        name = which
    t0 = time.time()
    base1, final, m1, m2, s2 = repeat(fp, tab, name, win, extra)
    rows = [row(f'{which}: round 1', base1, m1), row(f'{which}: round 1 + round 2', final, m2)]
    print(f'{time.time() - t0:.0f} s; round-2 moves: {len(s2)}')
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f'results_step3_repeat_{which}.csv', index=False)
    print(res.to_string(index=False))
