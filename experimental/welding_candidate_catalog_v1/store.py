"""Atomic SQLite storage for exhaustive sampled candidate trials."""
import json
import sqlite3
from pathlib import Path


def encode(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


class CatalogStore:
    def __init__(self,path,identity):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS relative(weld TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks(stage TEXT,weld TEXT,angle REAL,payload TEXT NOT NULL,PRIMARY KEY(stage,weld,angle));
        CREATE TABLE IF NOT EXISTS trials(stage TEXT,weld TEXT,angle REAL,robot TEXT,roll REAL,sign INTEGER,status TEXT,payload TEXT NOT NULL,PRIMARY KEY(stage,weld,angle,robot,roll,sign));
        CREATE TABLE IF NOT EXISTS candidates(weld TEXT,angle REAL,robot TEXT,roll REAL,sign INTEGER,branch INTEGER,payload TEXT NOT NULL,PRIMARY KEY(weld,angle,robot,roll,sign,branch));
        ''')
        existing=self.db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if existing and existing[0]!=encode(identity):
            self.db.close();raise ValueError('Catalogue input/code identity mismatch; use a new output directory')
        if not existing:
            with self.db:self.db.execute("INSERT INTO meta VALUES('identity',?)",(encode(identity),))
        self.identity=identity

    def close(self):
        self.db.close()

    def relative(self,weld):
        row=self.db.execute('SELECT payload FROM relative WHERE weld=?',(weld,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_relative(self,weld,payload):
        with self.db:self.db.execute('INSERT OR REPLACE INTO relative VALUES(?,?)',(weld,encode(payload)))

    def has_angle(self,weld,angle,stage='ik'):
        return self.db.execute('SELECT 1 FROM tasks WHERE stage=? AND weld=? AND angle=?',(stage,weld,angle)).fetchone() is not None

    def angle_rows(self,weld,angle,stage='fixed'):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM trials WHERE stage=? AND weld=? AND angle=? ORDER BY robot,roll,sign',(stage,weld,angle))]

    def save_angle(self,weld,angle,rows,stage='ik',metadata=None):
        if stage not in ('fixed','ik'):raise ValueError('Unknown stage')
        with self.db:
            self.db.execute('DELETE FROM trials WHERE stage=? AND weld=? AND angle=?',(stage,weld,angle))
            if stage=='ik':self.db.execute('DELETE FROM candidates WHERE weld=? AND angle=?',(weld,angle))
            for row in rows:
                pose=(row['robot_id'],row['roll_deg'],row['axis_sign'])
                self.db.execute('INSERT INTO trials VALUES(?,?,?,?,?,?,?,?)',(stage,weld,angle,*pose,row['status'],encode(row)))
                if stage=='ik':
                    for branch,state in enumerate(row.get('accepted',[])):
                        self.db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?)',(weld,angle,*pose,branch,encode(state)))
            self.db.execute('INSERT OR REPLACE INTO tasks VALUES(?,?,?,?)',(stage,weld,angle,encode(metadata or {})))

    def summary(self):
        count=lambda table:int(self.db.execute('SELECT count(*) FROM '+table).fetchone()[0])
        return dict(relative_welds=count('relative'),angle_tasks=count('tasks'),
            fixed_angle_tasks=int(self.db.execute("SELECT count(*) FROM tasks WHERE stage='fixed'").fetchone()[0]),
            ik_angle_tasks=int(self.db.execute("SELECT count(*) FROM tasks WHERE stage='ik'").fetchone()[0]),
            pose_trials=count('trials'),accepted_configurations=count('candidates'),
            statuses=[dict(stage=r[0],status=r[1],count=r[2]) for r in self.db.execute('SELECT stage,status,count(*) FROM trials GROUP BY stage,status')],
            welds_with_candidates=int(self.db.execute('SELECT count(DISTINCT weld) FROM candidates').fetchone()[0]))
