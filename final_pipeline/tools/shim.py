"""Drop-in stand-ins for librosa.midi_to_hz and mir_eval.transcription.match_notes (same definition as mir_eval 0.7:
onset within tol, pitch within cents tol, optional offset within max(ratio*ref_dur, min_tol); count = max bipartite matching)."""
import sys, types, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
def midi_to_hz(m): return 440.0*2.0**((np.asarray(m,float)-69)/12)
def match_notes(ref_int,ref_pitch,est_int,est_pitch,onset_tolerance=.05,pitch_tolerance=50.,offset_ratio=.2,offset_min_tolerance=.05,strict=False):
    on=np.abs(ref_int[:,0,None]-est_int[None,:,0])<=onset_tolerance
    pit=np.abs(1200*np.log2(est_pitch[None,:]/ref_pitch[:,None]))<=pitch_tolerance
    ok=on&pit
    if offset_ratio is not None:
        tol=np.maximum(offset_ratio*(ref_int[:,1]-ref_int[:,0]),offset_min_tolerance)
        ok&=np.abs(ref_int[:,1,None]-est_int[None,:,1])<=tol[:,None]
    if not ok.any(): return []
    m=maximum_bipartite_matching(csr_matrix(ok.astype(np.int8)),perm_type='column')
    return [(i,int(c)) for i,c in enumerate(m) if c>=0]
lib=types.ModuleType('librosa'); lib.midi_to_hz=midi_to_hz; sys.modules['librosa']=lib
me=types.ModuleType('mir_eval'); tr=types.ModuleType('mir_eval.transcription'); tr.match_notes=match_notes; me.transcription=tr
sys.modules['mir_eval']=me; sys.modules['mir_eval.transcription']=tr
