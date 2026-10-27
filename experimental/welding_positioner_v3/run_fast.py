"""Safe public entry: a completed benchmark is never overwritten by resume."""
import json
from pathlib import Path


def completed_run(folder):
    path = Path(folder)/'report.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result.get('phase') == 'complete':
            return result
    return None


if __name__ == '__main__':
    import fast_search
    result = completed_run(fast_search.OUT)
    if result is not None:
        print('Search already completed. Original results and timing preserved.', result['summary'])
    else:
        fast_search.main()
