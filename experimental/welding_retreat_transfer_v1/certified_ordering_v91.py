"""Classical ordering costs prioritizing accepted connections over untried edges."""
import numpy as np


def proposal_costs(original, known, penalties):
    costs=np.array(original,dtype=float,copy=True)
    finite=costs[np.isfinite(costs)]
    surcharge=(len(costs)+1)*(float(np.max(finite)) if len(finite) else 1.)
    costs+=surcharge
    for (i,j),edge in known.items():
        costs[i,j]=edge['metric_length']+penalties[i,j] if edge['passed'] else np.inf
    np.fill_diagonal(costs,np.inf)
    return costs
