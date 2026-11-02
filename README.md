# Welding Motion Planner

## Reproduction release (2026-11-02)

Use the **v3 source ZIP**, raw industrial dataset, checksum list and guide from:

https://github.com/173hyj/welding-motion-planner/releases/tag/reproduction-data-20261102

The release contains all source and original models required by the documented cold300 pipeline. Follow REPRODUCE_FROM_RAW_21_WELDS.md.

Verified raw-input run: 21/21 new 300 mm actions, 20/20 transfers, 63,274 trajectory states, **402.24 s**. Tests: 148 passed, 21 skipped.

The historical **349.83 s** optimized route is a separate sealed reference. Its full optimization history has **not** been reproduced from raw inputs. Do not treat reference playback as a cold planning rerun.

Some newer fixes could not be pushed through Git transport. Use the complete v3 release source snapshot rather than assuming this main branch contains every fix.

Source code: MIT. Third-party model rights remain separate.
