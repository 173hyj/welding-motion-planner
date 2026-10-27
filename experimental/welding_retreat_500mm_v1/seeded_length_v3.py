"""Bounded old route proposals; every reused proposal is freshly certified."""
import sys
from pathlib import Path
import numpy as np
from . import per_length_fast

def main():
    from experimental.welding_full_candidate_selection_v1 import raw_transfer
    from experimental.welding_candidate_selection_v1.edges import compress_collinear
    path=Path(sys.argv[sys.argv.index('--incumbent')+1]);old=np.load(path,allow_pickle=False)['states']
    original=raw_transfer.make_transfer
    def proposed(world,bridge,arm,angle_deg,start7,goal7,existing_edges=(),seed=0,**kwargs):
        direct_kwargs=dict(kwargs,max_iterations=0,enable_gateways=False)
        direct,stats=original(world,bridge,arm,angle_deg,start7,goal7,existing_edges=(),seed=seed,**direct_kwargs)
        if direct is not None:return direct,stats
        axes=np.r_[np.arange(arm*6,arm*6+6),13+arm]
        matches=np.flatnonzero(np.isclose(old[:,12],np.deg2rad(angle_deg),atol=1e-12,rtol=0))
        extra=[]
        if len(matches):
            local=old[matches][:,axes];scale=np.r_[np.ones(6),100.]
            da=np.linalg.norm((local-start7)/scale,axis=1);db=np.linalg.norm((local-goal7)/scale,axis=1)
            ia=int(matches[np.argmin(da)]);ib=int(matches[np.argmin(db)])
            if min(da)<.3 and min(db)<.3 and abs(ib-ia)<100000:
                section=old[min(ia,ib):max(ia,ib)+1]
                if np.allclose(section[:,12],np.deg2rad(angle_deg),atol=1e-12,rtol=0):
                    local=section[:,axes]
                    if ia>ib:local=local[::-1]
                    candidate=np.vstack([start7,local,goal7])
                    candidate=candidate[np.r_[True,np.any(abs(np.diff(candidate,axis=0))>1e-12,axis=1)]]
                    candidate[0]=start7;candidate[-1]=goal7
                    if len(candidate)>128:candidate=candidate[np.unique(np.linspace(0,len(candidate)-1,128).astype(int))]
                    extra=[candidate]
        return original(world,bridge,arm,angle_deg,start7,goal7,existing_edges=extra+list(existing_edges),seed=seed,**kwargs)
    raw_transfer.make_transfer=proposed
    per_length_fast.main()
if __name__=='__main__':main()
