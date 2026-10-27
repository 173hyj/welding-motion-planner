import unittest
import numpy as np
from endpoint_identity import same_physical_endpoint


class EndpointIdentityTests(unittest.TestCase):
    def test_accepts_roundoff_from_scale_and_reverse(self):
        a=np.zeros(15);a[7]=-.11912803703066109;b=a.copy();b[7]=-.1191280370306611
        self.assertTrue(same_physical_endpoint(a,b))

    def test_rejects_material_joint_or_driver_displacement(self):
        a=np.zeros(15)
        for i in (0,5,12,13,14):
            b=a.copy();b[i]=1e-8
            self.assertFalse(same_physical_endpoint(a,b))

    def test_rejects_shape_or_nonfinite(self):
        self.assertFalse(same_physical_endpoint(np.zeros(15),np.zeros(14)))
        b=np.zeros(15);b[1]=np.nan
        self.assertFalse(same_physical_endpoint(np.zeros(15),b))


if __name__=='__main__':unittest.main()
