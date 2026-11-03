"""
Whole-simulation CPU benchmark of SPECULA: the open-loop SCAO scenario in
scao_multi.yml, with numpy, with the C++ backend, and with the C++ backend and
psf_dt = 5 ms on the four science cameras. Each configuration runs in its own
process; timings come from the trace summary of the simulation (--trace-file).
See README.md.

Usage: python cpp/benchmarks/bench_scao.py
"""
import os
import resource
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(HERE))
YML = os.path.join(HERE, 'scao_multi.yml')
SKIP = 5  # first iterations, not timed
PSF_DT = '{psf_ngs.psf_dt: 0.005, psf_sci1.psf_dt: 0.005, psf_sci2.psf_dt: 0.005, psf_sci3.psf_dt: 0.005}'
CONFIGS = {'numpy': (True, None), 'C++': (False, None), 'C++ psf_dt': (False, PSF_DT)}
PHASES = ('inputs', 'prepare_trigger', 'trigger', 'post_trigger')


def worker(config):
    no_cpp, overrides = CONFIGS[config]
    sys.path.insert(0, REPO_DIR)
    import specula
    with tempfile.TemporaryDirectory() as work_dir:
        os.chdir(work_dir)  # root_dir of the scenario (phase screens) and trace files
        os.mkdir('calib')
        specula.main_simul([YML], cpu=True, no_cpp=no_cpp, overrides=overrides,
                           trace_file='trace.tsv', trace_skip=SKIP,
                           no_speed_report=True, log_level='WARNING')
        with open('trace.summary.txt') as f:
            summary = f.read().splitlines()

    # Per-object time per step: all the phases of each object, over the timed iterations
    per_object = {}
    n_iter = None
    for line in summary:
        fields = line.split()
        if line.startswith('Iterations:'):
            n_iter = int(fields[1].rstrip(','))
            print(f'step_ms\t{float(line.split("(")[1].split()[0]) / 1000:.2f}')
        elif len(fields) == 8 and fields[2] in PHASES and fields[3].isdigit():
            per_object[fields[0]] = per_object.get(fields[0], 0.0) + float(fields[4])
    for name, total_ms in per_object.items():
        print(f'{name}\t{total_ms / n_iter:.2f}')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f'peak_mb\t{rss / 2**20 if sys.platform == "darwin" else rss / 2**10:.0f}')


def run(config):
    env = dict(os.environ, SPECULA_DISABLE_GPU='TRUE')
    env.pop('SPECULA_DISABLE_CPP', None)
    out = subprocess.run([sys.executable, __file__, '--worker', config], env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f'{config} run failed:\n{out.stderr}')
    return dict(line.split('\t') for line in out.stdout.splitlines() if '\t' in line)


def main():
    res = {config: run(config) for config in CONFIGS}
    base = float(res['numpy']['step_ms'])
    print(f'{"Configuration":14s} {"ms/step":>8s} {"speed-up":>9s} {"peak MB":>8s}')
    for config, r in res.items():
        step = float(r['step_ms'])
        print(f'{config:14s} {step:8.1f} {base / step:8.1f}x {r["peak_mb"]:>8s}')
    print()
    objects = sorted((k for k in res['numpy'] if k not in ('step_ms', 'peak_mb')),
                     key=lambda k: -float(res['numpy'][k]))
    print(f'{"Object (ms/step)":16s}' + ''.join(f'{c:>12s}' for c in CONFIGS))
    for name in objects:
        print(f'{name:16s}' + ''.join(f'{float(res[c].get(name, 0)):12.2f}' for c in CONFIGS))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        main()
