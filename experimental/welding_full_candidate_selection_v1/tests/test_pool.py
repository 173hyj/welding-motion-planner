import importlib.util
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('full_pool',Path(__file__).parents[1]/'pool.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PoolTests(unittest.TestCase):
    def test_all_catalog_rows_survive_registration_without_top_k_or_q_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'catalog.sqlite'
            with closing(sqlite3.connect(file)) as db:
                db.execute('CREATE TABLE candidates(angle REAL,weld TEXT,robot TEXT,roll REAL,sign INTEGER,branch INTEGER,payload TEXT)')
                for i in range(30):
                    db.execute('INSERT INTO candidates VALUES(120,\'w\',\'r\',?,?,?,?)',(i%2*90,1,i,json.dumps(dict(q_rad=[i*1e-8]*6,origin='numeric_retained'))))
                db.execute('INSERT INTO candidates VALUES(90,\'w\',\'r\',0,1,99,?)',(json.dumps(dict(q_rad=[0]*6)),))
                db.commit()
            layers=[dict(task=1,weld_id='w',angle=120,robot_id='r')]
            rows=p.register_catalog(file,layers)
            self.assertEqual(len(rows),30)
            self.assertEqual(len({row['id'] for row in rows}),30)
            self.assertEqual(rows,p.register_catalog(file,layers))
            self.assertTrue(all(row['status']=='pending' for row in rows))

    def test_coverage_distinguishes_pending_and_finite_unresolved(self):
        rows=[dict(status=s) for s in ['certified','pending','search_unresolved','proven_invalid']]
        c=p.coverage(rows)
        self.assertFalse(c['complete'])
        self.assertEqual(c['registered'],4)
        rows[1]['status']='search_unresolved'
        self.assertTrue(p.coverage(rows)['complete'])
        self.assertFalse(p.coverage(rows)['exhaustive_feasibility_search'])

    def test_retry_queue_preserves_row_identity(self):
        rows=[dict(id='a',status='certified'),dict(id='b',status='pending'),dict(id='c',status='search_unresolved')]
        self.assertEqual([r['id'] for r in p.pending_rows(rows)],['b'])
        self.assertEqual([r['id'] for r in p.pending_rows(rows,True)],['b','c'])


if __name__=='__main__':unittest.main()
