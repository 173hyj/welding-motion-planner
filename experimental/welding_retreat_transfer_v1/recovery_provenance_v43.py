"""Explicit new evidence chain after dependency loss; never repair old seals by omission."""


def classify_seals(seals,digest_or_none):
    matched,missing = {},{}
    for name,expected in seals.items():
        actual = digest_or_none(name)
        if actual is None:
            missing[name] = expected
        elif actual != expected:
            raise ValueError('Changed existing source: '+name)
        else:
            matched[name] = expected
    return dict(matched=matched,missing=missing,historical_chain_intact=not missing,
                requires_fresh_cold=True)


def validate_inventory(rows,distance):
    if distance != 2000. or len(rows)!=21 or len({r['weld_id'] for r in rows})!=21:
        raise ValueError('Complete original 21-weld inventory and 2000mm required')
