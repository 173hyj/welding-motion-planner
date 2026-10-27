"""Tie-aware immutable source subsection proposals, freshly fine certified."""
import sys
import numpy as np
from pathlib import Path
from . import per_length_fast

def select_subsection(old,phases,arm,angle,start,goal,park):
 axes=np.r_[np.arange(arm*6,arm*6+6),13+arm];fixed=np.setdiff1d(np.arange(15),axes)
 base=np.zeros(15);base[12]=np.deg2rad(angle);base[6*(1-arm):6*(1-arm)+6]=park[:6];base[13+1-arm]=park[6]
 eligible=np.all(np.isclose(old[:,fixed],base[fixed],atol=1e-12,rtol=0),axis=1)&~np.isin(phases[:,arm],['weld','close','open'])
 ids=np.flatnonzero(eligible)
 if not len(ids):return None
 scale=np.r_[np.ones(6),100.];local=old[:,axes]
 da=np.linalg.norm((local-start)/scale,axis=1);db=np.linalg.norm((local-goal)/scale,axis=1)
 amin=da[ids].min();bmin=db[ids].min()
 if max(amin,bmin)>.3:return None
 starts=set(np.flatnonzero(eligible&(da<=amin+1e-8)).tolist());goals=set(np.flatnonzero(eligible&(db<=bmin+1e-8)).tolist())
 best=None
 for run in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
  aa=[int(i) for i in run if int(i) in starts];bb=np.array([int(i) for i in run if int(i) in goals])
  if not len(bb):continue
  for a in aa:
   b=int(bb[np.argmin(abs(bb-a))]);score=(abs(b-a),a,b)
   if best is None or score<best:best=score
 if best is None:return None
 _,a,b=best;section=old[min(a,b):max(a,b)+1]
 return section.copy() if a<=b else section[::-1].copy()

def main():
 from experimental.welding_full_candidate_selection_v1 import raw_transfer
 from experimental.welding_candidate_selection_v1.edges import compress_collinear
 z=np.load(Path(sys.argv[sys.argv.index('--incumbent')+1]),allow_pickle=False);old=z['states'];phases=z['phases'];original=raw_transfer.make_transfer
 def proposed(world,bridge,arm,angle,start,goal,existing_edges=(),seed=0,**kwargs):
  direct,stats=original(world,bridge,arm,angle,start,goal,existing_edges=(),seed=seed,**dict(kwargs,max_iterations=0,enable_gateways=False))
  if direct is not None:return direct,stats
  section=select_subsection(old,phases,arm,angle,start,goal,np.asarray(kwargs['park7']))
  extra=[]
  if section is not None:
   axes=np.r_[np.arange(arm*6,arm*6+6),13+arm]
   section=compress_collinear(section)[:,axes]
   candidate=np.vstack([start,section,goal]);candidate=candidate[np.r_[True,np.any(abs(np.diff(candidate,axis=0))>1e-12,axis=1)]]
   candidate[0]=start;candidate[-1]=goal;extra=[candidate]
   print('EXACT_SOURCE_PREFIX_PROPOSAL',arm,angle,len(candidate),flush=True)
  return original(world,bridge,arm,angle,start,goal,existing_edges=extra+list(existing_edges),seed=seed,**kwargs)
 raw_transfer.make_transfer=proposed;per_length_fast.main()
if __name__=='__main__':main()
