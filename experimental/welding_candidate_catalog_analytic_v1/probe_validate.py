"""Validate source-probe random legal FK targets; never modifies source inputs."""
import json
import sys
from pathlib import Path
import numpy as np
from welding_kinematics import RobotKinematics
from .analytic import FanucAnalytic


def validate(probe_path,output_path):
    probe=json.loads(Path(probe_path).read_text(encoding='utf-8'))
    rng=np.random.default_rng(7934); rows=[]
    for asset in probe['assets']:
        if 'robot' not in asset: continue
        model=RobotKinematics({'joints':probe['joints']},asset)
        solver=FanucAnalytic(model); maximum=0.; legal=0; total=0
        while legal<100:
            q=rng.uniform(*model.bounds); total+=1
            if not model.limits_valid(q): continue
            legal+=1
            solutions,stats=solver.solve(model.tcp(q))
            error=min((float(np.linalg.norm(np.asarray(s['q_rad'])-q)) for s in solutions),default=float('inf'))
            maximum=max(maximum,error)
            if error>1e-6: raise AssertionError((asset['name'],q,error,stats))
        rows.append(dict(robot=asset['name'],random_legal_targets=legal,sampled_configurations=total,
                         max_recovered_joint_error_rad=maximum,metadata=solver.metadata))
    result=dict(passed=True,robots=rows)
    Path(output_path).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='robots'}),[(r['robot'],r['max_recovered_joint_error_rad']) for r in rows])


if __name__=='__main__': validate(sys.argv[1],sys.argv[2])
