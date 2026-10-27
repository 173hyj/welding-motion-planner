"""Adapt all certified catalog actions and remap sealed cached graph identities."""
import argparse
import copy
import json
from pathlib import Path
from experimental.welding_candidate_selection_v1.pool import atomic,digest
from experimental.welding_candidate_selection_v1.run import verify_bindings


def remap_cached_edges(old_layers,new_layers,rows):
    def index(task,old_index):
        checksum=old_layers[task]['actions'][old_index]['path_sha256']
        if old_index==0:
            if new_layers[task]['actions'][0]['path_sha256']!=checksum:raise ValueError('Baseline candidate identity changed')
            return 0
        matches=[i for i,a in enumerate(new_layers[task]['actions']) if a['path_sha256']==checksum]
        if not matches:raise ValueError('Cached action absent from expanded layer')
        return matches[0]
    result=[]
    for row in rows:
        k,i,j=row['key'];mapped=copy.deepcopy(row)
        if k==-1:j=index(7,j)
        elif k==13:i=index(20,i)
        else:i,j=index(k+7,i),index(k+8,j)
        mapped['key']=[k,i,j];mapped['original_key']=row['key'];result.append(mapped)
    return result


def prepare(full_pool,old_pool,previous,out):
    files=[Path(folder)/'report.json' for folder in (full_pool,old_pool,previous)]
    full,old,prior=[json.loads(p.read_text()) for p in files];out=Path(out).resolve()
    if out.exists():raise FileExistsError('Fresh adapter output required')
    if not full.get('complete') or len(full['layers'])!=21:raise ValueError('Complete all21 catalog pool required')
    if prior['pool_sha256']!=digest(files[1]):raise ValueError('Cached graph and original pool differ')
    bindings=dict(prior['manifest']['bindings'])
    for field in ('bindings','source_sha256','code_sha256'):bindings.update(full['manifest'].get(field,{}))
    bindings.update({str(p.resolve()):digest(p) for p in files})
    own=Path(__file__).absolute();bindings[str(own)]=digest(own)
    layers=copy.deepcopy(full['layers']);supplemental=[]
    for old_layer,new_layer in zip(old['layers'],layers):
        if (old_layer['task'],old_layer['weld_id'],old_layer['angle'],old_layer['arm'])!=(new_layer['task'],new_layer['weld_id'],new_layer['angle'],new_layer['arm']):raise ValueError('Layer fixed identity changed')
        baseline=old_layer['actions'][0]
        if new_layer['actions'][0]['path_sha256']!=baseline['path_sha256']:raise ValueError('Baseline action changed')
        known={a['path_sha256'] for a in new_layer['actions']}
        for action in old_layer['actions'][1:]:
            if action['path_sha256'] not in known:
                new_layer['actions'].append(copy.deepcopy(action));known.add(action['path_sha256']);supplemental.append(dict(task=old_layer['task'],action_id=action['id'],path_sha256=action['path_sha256']))
        for action in new_layer['actions']:bindings[action['path_file']]=action['path_sha256']
    mapped=remap_cached_edges(old['layers'],layers,prior['edges'])
    verify_bindings(bindings);verify_bindings({r['file']:r['sha256'] for r in mapped if r.get('passed')})
    adapter=copy.deepcopy(full);adapter.update(layers=layers,prefix_end_index=old['prefix_end_index'],baseline_path=old['baseline_path'],baseline_sha256=old['baseline_sha256'],supplemental_cached_actions=supplemental,adapter_policy='All full-pool certified nodes plus exact previously certified cached actions; first seven frozen only in this optional tail comparison.',full_pool_report=str(files[0].resolve()),full_pool_sha256=digest(files[0]))
    adapter['manifest']=dict(bindings=bindings,source_sha256=bindings.copy(),code_sha256={str(own):digest(own)})
    out.mkdir();atomic(out/'pool/report.json',adapter)
    cached=copy.deepcopy(prior);cached['edges']=mapped;cached['pool_sha256']=digest(out/'pool/report.json');cached['manifest']['bindings']=dict(bindings,**{str((out/'pool/report.json').resolve()):cached['pool_sha256']})
    cached['adapter_provenance']=dict(original_report=str(files[2].resolve()),original_report_sha256=digest(files[2]),action_matching='exact_path_sha256_baseline_zero')
    atomic(out/'previous/report.json',cached);verify_bindings(bindings)
    return dict(pool=str(out/'pool'),previous=str(out/'previous'),all_action_count=sum(len(x['actions']) for x in layers),tail_action_count=sum(len(x['actions']) for x in layers[7:]),supplemental_cached_actions=supplemental)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('full-pool','old-pool','previous','out'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();print(json.dumps(prepare(args.full_pool,args.old_pool,args.previous,args.out),indent=2))

if __name__=='__main__':main()
