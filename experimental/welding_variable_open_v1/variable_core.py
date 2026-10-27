"""Opening-prefix contract: a later clear state cannot bypass a blocked state."""
import math
import numpy as np


def driver_samples(driver_mm, step_mm=1.):
    driver_mm = float(driver_mm)
    if not math.isfinite(driver_mm) or not -185. <= driver_mm <= 0.:
        raise ValueError('Opening driver must be finite in [-185,0] mm')
    if not math.isfinite(step_mm) or step_mm <= 0.:
        raise ValueError('Positive finite sampling step required')
    count = max(1, int(math.ceil(abs(driver_mm)/step_mm)))
    return np.linspace(0., driver_mm, count+1) if driver_mm else np.array([0.])


def opening_profile(probe, electrode_ids, maximum_amplitude_mm=185.):
    passed_drivers = []; failure = None; checked = 0
    for driver in driver_samples(-maximum_amplitude_mm):
        driver = float(driver); state = probe(driver); checked += 1
        certified = driver != 0. or all(state.get('certificates', {}).get(p, {}).get('passed', False) for p in electrode_ids)
        if not state['passed'] or not certified:
            failure = dict(driver_mm=driver, state=state, closed_electrodes_certified=certified)
            break
        passed_drivers.append(driver)
    return dict(checked_samples=checked, passed_drivers_mm=passed_drivers,
                last_passed_amplitude_mm=abs(passed_drivers[-1]) if passed_drivers else None,
                initial_certified=bool(passed_drivers), failure=failure)


def prefix_available(profile, amplitude_mm):
    driver_samples(-amplitude_mm)
    return profile['initial_certified'] and amplitude_mm <= profile['last_passed_amplitude_mm']+1e-10


def completed_report(path, manifest, *, resume):
    import json
    from pathlib import Path
    path = Path(path)
    if not path.exists(): return None
    if not resume: raise FileExistsError('Evidence exists; use --resume or a new output directory')
    report = json.loads(path.read_text())
    if report['manifest'] != manifest:
        raise RuntimeError('Input/code/settings differ; refusing to overwrite evidence')
    return report if report['summary']['complete'] else None
