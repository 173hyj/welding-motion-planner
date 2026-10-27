"""Consume one passing short distance without accepting another failed distance."""
import sys
from . import per_length_fast,seeded_length_v3

def require_selected_generation(report,length):
 if length not in (200,300):raise ValueError('Approved short length required')
 if report.get('source_assets_frozen_before_and_after') is not True or report.get('complete') is not True:raise ValueError('Complete final source seal required')
 rows=report['rows']
 if len(rows)!=42 or {(r['length_mm'],r['task']) for r in rows}!={(d,t) for d in (200.,300.) for t in range(1,22)}:raise ValueError('Exact 42 fresh actions required')
 selected=[r for r in rows if r['length_mm']==length]
 if len(selected)!=21 or not all(r['passed'] for r in selected):raise ValueError('Selected distance has rejected action')
 return rows

def main():
 length=int(sys.argv[sys.argv.index('--length')+1])
 per_length_fast.require_generation=lambda report:require_selected_generation(report,length)
 seeded_length_v3.main()
if __name__=='__main__':main()
