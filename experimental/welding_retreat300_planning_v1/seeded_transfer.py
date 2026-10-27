"""Bounded direct-first free-motion incumbent seeds; never crosses welds."""
import numpy as np
OLD_STATES=None
OLD_PHASES=None
REFERENCES=[]

def make_transfer(world,bridge,arm,angle,start,goal,existing_edges=(),seed=0,**kwargs):
    from experimental.welding_full_candidate_selection_v1.raw_transfer import make_transfer as original
    from experimental.welding_retreat_250mm_v1.exact_prefix_v4 import select_subsection
    from experimental.welding_candidate_selection_v1.edges import compress_collinear
    direct,stats=original(world,bridge,arm,angle,start,goal,existing_edges=(),seed=seed,**dict(kwargs,max_iterations=0,enable_gateways=False))
    if direct is not None:return direct,stats
    extra=[]
    references=list(REFERENCES)
    if OLD_STATES is not None:references.insert(0,(OLD_STATES,OLD_PHASES))
    for old,phases in references:
        section=select_subsection(old,phases,arm,angle,start,goal,np.asarray(kwargs['park7']))
        if section is not None:
            axes=np.r_[np.arange(arm*6,arm*6+6),13+arm]
            section=compress_collinear(section)[:,axes]
            proposal=np.vstack([start,section,goal])
            proposal=proposal[np.r_[True,np.any(abs(np.diff(proposal,axis=0))>1e-12,axis=1)]]
            proposal[0]=start;proposal[-1]=goal;extra.append(proposal)
    return original(world,bridge,arm,angle,start,goal,existing_edges=extra+list(existing_edges),seed=seed,**kwargs)
