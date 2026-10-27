"""Physical 15-axis identity, allowing only floating point scale roundoff."""
import numpy as np


def same_physical_endpoint(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return bool(a.shape==(15,) and b.shape==(15,) and np.isfinite(a).all()
                and np.isfinite(b).all() and np.allclose(a,b,atol=1e-12,rtol=0.))
