"""Monotone sequential-QP controller; acceptance relies on the full evaluator."""


def optimize_iterations(initial,step,evaluate,*,max_iterations=12):
    state=initial;current=evaluate(state);history=[];reason='iteration_limit'
    if current['passed']:
        return dict(state=state,evaluation=current,passed=True,iterations=0,history=[],reason='initial_passed')
    for iteration in range(max_iterations):
        update=step(state);record=dict(iteration=iteration+1,update=update,candidates=[])
        if not update['solved']:
            history.append(record);reason='qp_not_solved';break
        best=None
        for proposal in update['proposals']:
            result=evaluate(proposal);record['candidates'].append(dict(state=proposal,evaluation=result))
            if result['passed']:
                best=(proposal,result);break
            if result['score']>current['score'] and (best is None or result['score']>best[1]['score']):best=(proposal,result)
        if best is None:
            history.append(record);reason='no_improving_step';break
        state,current=best;record['accepted_state']=state;history.append(record)
        if current['passed']:reason='full_path_passed';break
    return dict(state=state,evaluation=current,passed=current['passed'],iterations=len(history),history=history,reason=reason)
