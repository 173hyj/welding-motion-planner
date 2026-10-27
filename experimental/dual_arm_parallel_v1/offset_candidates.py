"""Pure time-shift proposals; collision acceptance remains the runner's job."""
import math


def offset_candidates(chains, base_offsets=(), weld_seconds=.6):
    """Rank offset records by makespan, then simultaneous welding.

    ``chains`` holds two motion chains with ``times`` and ``pieces``.
    Generate weld start/end alignments and full-overlap boundaries in both
    shift directions, duration-slack boundaries, baseline and serial offsets.
    Existing search offsets may be supplied in ``base_offsets``.

    Records with the same makespan prefer a single continuous simultaneous
    weld interval of at least ``weld_seconds``, then total weld overlap.
    Checking in this order and selecting the first fine-valid record preserves
    minimum makespan *among these candidates*, not over all real-valued shifts.
    No record has collision certification merely from being listed here.
    """
    if len(chains)!=2 or not math.isfinite(weld_seconds) or weld_seconds<=0:
        raise ValueError('two chains and positive finite weld_seconds required')
    durations=[]; intervals=[]
    for chain in chains:
        duration=float(chain['times'][-1])
        if not math.isfinite(duration) or duration<0:
            raise ValueError('finite nonnegative chain duration required')
        durations.append(duration)
        welds=[]
        for piece in chain['pieces']:
            if piece['phase']!='weld':continue
            start,end=float(piece['start']),float(piece['end'])
            if not (math.isfinite(start) and math.isfinite(end) and 0<=start<=end<=duration+1e-9):
                raise ValueError('weld interval outside chain duration')
            if end>start:welds.append((start,end))
        welds.sort()
        # Merge overlapping/adjacent intervals to avoid double-counting.
        merged=[]
        for start,end in welds:
            if merged and start<=merged[-1][1]+1e-12:
                merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
            else:merged.append((start,end))
        intervals.append(merged)
    candidates=list(base_offsets)+[(0.,0.),(0.,durations[0]),(durations[1],0.),
                                  (0.,max(0.,durations[0]-durations[1])),
                                  (max(0.,durations[1]-durations[0]),0.)]
    for s0,e0 in intervals[0]:
        for s1,e1 in intervals[1]:
            # Relative delay d=offset1-offset0. Endpoint differences are the
            # breakpoints of the piecewise-linear overlap function; threshold
            # boundaries capture the earliest delay achieving a complete weld.
            delays=[s0-s1,e0-e1,s0-e1,e0-s1,
                    s0+weld_seconds-e1,e0-weld_seconds-s1]
            for delay in delays:
                candidates.append((max(0.,-delay),max(0.,delay)))
    rows=[];seen=set()
    for candidate in candidates:
        offsets=tuple(float(v) for v in candidate)
        if len(offsets)!=2 or any(not math.isfinite(v) or v<0 for v in offsets):
            raise ValueError('two finite nonnegative offsets required')
        if offsets in seen:continue
        seen.add(offsets)
        overlap=[]
        for s0,e0 in intervals[0]:
            for s1,e1 in intervals[1]:
                start=max(s0+offsets[0],s1+offsets[1])
                end=min(e0+offsets[0],e1+offsets[1])
                if end>start:overlap.append(end-start)
        maximum=max(overlap,default=0.)
        rows.append(dict(offsets=offsets,
                         makespan=max(durations[a]+offsets[a] for a in (0,1)),
                         total_weld_overlap_seconds=sum(overlap),
                         max_weld_overlap_seconds=maximum,
                         full_weld_overlap=maximum>=weld_seconds-1e-9))
    return sorted(rows,key=lambda r:(r['makespan'],not r['full_weld_overlap'],
                                    -r['total_weld_overlap_seconds'],
                                    -r['max_weld_overlap_seconds'],sum(r['offsets']),r['offsets']))
