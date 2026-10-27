import ast
import itertools
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).absolute().parents[1]
ns={'np':np,'itertools':itertools}
for filename,names in [('transfers.py',{'order_nodes','order_batches'}),('merge_actions.py',{'cover_order'})]:
    tree=ast.parse((HERE/filename).read_text(encoding='utf-8'))
    body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body,type_ignores=[]),filename,'exec'),ns)


class RotationOrderTests(unittest.TestCase):
    def test_cover_keeps_start_and_end_stops_to_minimize_rotations(self):
        route=ns['cover_order']([(0.,30.,60.,90.,120.)],30.,90.)
        self.assertEqual(route[0],30.)
        self.assertEqual(route[-1],90.)
        self.assertEqual(sum(abs(b-a) for a,b in zip((30.,)+route,route+(90.,))),180.)

    def test_batches_keep_each_angle_contiguous(self):
        states=[]
        for a in [0,30,60,90,120,90,30]:
            s=np.zeros(15);s[12]=np.deg2rad(a);states.append(s)
        start=np.zeros(15);start[12]=np.deg2rad(30)
        end=np.zeros(15);end[12]=np.deg2rad(90)
        route=ns['order_batches'](states,start,end)
        self.assertEqual(sorted(route),list(range(7)))
        angles=[round(np.rad2deg(states[i][12])) for i in route]
        distinct=[a for i,a in enumerate(angles) if i==0 or a!=angles[i-1]]
        self.assertEqual(len(distinct),5)
        self.assertEqual(distinct[0],30)
        self.assertEqual(distinct[-1],90)


if __name__=='__main__':unittest.main()
