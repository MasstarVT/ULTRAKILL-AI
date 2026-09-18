"""The twelve `SubprocVecEnv` workers must stay LIGHT, and the helper processes with them.

Measured 2026-09-18, the day this machine ran out of committed memory for the third time: each worker held
**0.91 GB of committed private bytes for 0.2 GB of working set**, and `dashboard.py`, `campaign_driver.py`
and `post_times.py` held 0.75 GB each for 0.04 GB. Two causes, and the smaller one is the famous one:

  * `import numpy` alone costs **785 MB of commit** (13 MB of working set) because OpenBLAS reserves a
    per-thread buffer pool at load -- ~34 MB per thread over 24 logical CPUs. `OPENBLAS_NUM_THREADS=1`
    takes it to 9 MB, and nothing in this project multiplies a matrix big enough to notice.
  * a spawn child re-executes the parent's `__main__` module, so `scripts/train.py`'s top-level
    `import torch` / `from stable_baselines3 import PPO` was paid by every worker: ~175 MB more each.

These tests pin both, in real subprocesses, because both are *import-order* properties that cannot be
observed from inside a process that has already imported numpy. No game and no network.

    python tests/test_light_workers.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultrakill_ai.procmem import BLAS_THREAD_VARS, cap_blas_threads  # noqa: E402

TRAIN_PY = ROOT / "scripts" / "train.py"
MB = 1e6

# What a `SubprocVecEnv` worker does on its way up, in order: multiprocessing's `_fixup_main_from_path`
# runs the parent's __main__ file under the name `__mp_main__`, and only then is the env_fn unpickled
# (which imports the modules the pickled closure names). This reproduces exactly that sequence.
WORKER_BOOT = """
import runpy, sys
sys.path.insert(0, r"{root}")
runpy.run_path(r"{train}", run_name="__mp_main__")
import ultrakill_ai.env, ultrakill_ai.envfactory
"""

PROBE_TAIL = """
import ctypes, ctypes.wintypes as wt, json, os, sys
class _MC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]
_c = _MC(); _c.cb = ctypes.sizeof(_c)
ctypes.windll.psapi.GetProcessMemoryInfo(wt.HANDLE(ctypes.windll.kernel32.GetCurrentProcess()),
                                         ctypes.byref(_c), _c.cb)
print("@@" + json.dumps({
    "modules": sorted({m.split(".")[0] for m in sys.modules}
                      & {"torch", "stable_baselines3", "sb3_contrib", "numpy", "gymnasium", "pandas"}),
    "blas": os.environ.get("OPENBLAS_NUM_THREADS"),
    "private": _c.PrivateUsage, "working_set": _c.WorkingSetSize}))
"""


def probe(body: str) -> dict:
    """Runs `body` in a FRESH interpreter and reports what it imported and what it cost."""
    script = textwrap.dedent(body) + PROBE_TAIL
    env = dict(os.environ)
    env.pop("OPENBLAS_NUM_THREADS", None)  # never let this process's own cap leak into the measurement
    env["PYTHONPATH"] = str(ROOT)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                          cwd=str(ROOT), timeout=300)
    assert done.returncode == 0, f"probe failed:\n{done.stdout}\n{done.stderr}"
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("@@")][-1]
    return json.loads(line[2:])


# --------------------------------------------------------------------------- the worker


def test_a_spawn_worker_imports_neither_torch_nor_stable_baselines3():
    """The headline property. A worker re-executes scripts/train.py; that must not cost it a deep learning stack."""
    result = probe(WORKER_BOOT.format(root=ROOT, train=TRAIN_PY))
    assert "torch" not in result["modules"], f"a worker imported torch: {result['modules']}"
    assert "stable_baselines3" not in result["modules"], f"a worker imported SB3: {result['modules']}"
    print(f"[ok] spawn worker imports {result['modules']}")


def test_a_spawn_worker_caps_blas_threads_before_numpy_is_imported():
    """The cap has to be applied by the __mp_main__ re-execution, which runs before anything unpickles numpy."""
    result = probe(WORKER_BOOT.format(root=ROOT, train=TRAIN_PY))
    assert result["blas"] == "1", f"OPENBLAS_NUM_THREADS was {result['blas']!r} in a worker"
    assert "numpy" in result["modules"], "the probe did not actually reach numpy, so it proves nothing"
    print("[ok] worker capped OPENBLAS_NUM_THREADS=1 with numpy loaded")


def test_a_worker_costs_far_less_than_the_measured_gigabyte():
    """The regression bar. Measured before this change: 910 MB. After: ~60 MB. 300 MB is the tripwire."""
    light = probe(WORKER_BOOT.format(root=ROOT, train=TRAIN_PY))
    heavy = probe(f"""
        import sys
        sys.path.insert(0, r"{ROOT}")
        import torch, numpy
        from stable_baselines3 import PPO
        import ultrakill_ai.env
    """)
    print(f"[measured] worker now {light['private'] / MB:.0f} MB private "
          f"({light['working_set'] / MB:.0f} MB working set); "
          f"the old worker's import set costs {heavy['private'] / MB:.0f} MB")
    assert light["private"] < 300 * MB, f"a worker costs {light['private'] / MB:.0f} MB of commit"
    assert light["private"] < heavy["private"] / 3, "the light worker should be a fraction of the old one"


def test_the_pickled_env_fn_unpickles_without_torch():
    """SB3 cloudpickles the env_fn. Every global the closure names must resolve inside a LIGHT module.

    This is the property that actually bites: a closure is pickled by value, but the names it references are
    pickled by reference, so wrapping the env in `stable_baselines3...Monitor` used to drag SB3 into the child
    no matter how light `train.py` was.
    """
    import cloudpickle

    from ultrakill_ai.env import EnvConfig
    from ultrakill_ai.envfactory import make_env

    blob = ROOT / "tests" / "_env_fn.pickle.tmp"
    blob.write_bytes(cloudpickle.dumps(make_env(EnvConfig(mode="campaign", port=47899), ("kills",))))
    try:
        result = probe(f"""
            import pickle, sys
            sys.path.insert(0, r"{ROOT}")
            fn = pickle.loads(open(r"{blob}", "rb").read())
            assert callable(fn)
        """)
    finally:
        blob.unlink(missing_ok=True)
    assert "torch" not in result["modules"], f"unpickling the env_fn imported {result['modules']}"
    assert "stable_baselines3" not in result["modules"], f"unpickling the env_fn imported {result['modules']}"
    print(f"[ok] env_fn unpickles with {result['modules']}")


# --------------------------------------------------------------------------- the package and the helpers


def test_importing_the_package_does_not_import_the_env():
    """`ultrakill_ai.times` used to cost 793 MB because the package __init__ imported `env` eagerly."""
    result = probe(f"""
        import sys
        sys.path.insert(0, r"{ROOT}")
        import ultrakill_ai
    """)
    assert result["modules"] == [], f"importing the package pulled in {result['modules']}"
    assert result["private"] < 40 * MB, f"importing the package cost {result['private'] / MB:.0f} MB"
    print(f"[ok] import ultrakill_ai costs {result['private'] / MB:.0f} MB and imports nothing heavy")


def test_the_lazy_package_still_exports_its_public_names():
    """PEP 562 laziness must not break `from ultrakill_ai import UltrakillEnv`."""
    import ultrakill_ai

    from ultrakill_ai.env import UltrakillEnv as direct

    assert ultrakill_ai.UltrakillEnv is direct
    assert ultrakill_ai.BridgeTimeout.__name__ == "BridgeTimeout"
    assert isinstance(ultrakill_ai.RECOVERABLE, tuple)
    assert "UltrakillEnv" in dir(ultrakill_ai)
    try:
        ultrakill_ai.NoSuchThing
    except AttributeError:
        pass
    else:
        raise AssertionError("a missing attribute must still raise AttributeError")
    print("[ok] lazy package exports its public names")


def test_no_helper_process_imports_torch_and_every_one_caps_blas():
    """The processes that run beside a trainer: a watchdog, a driver, a dashboard, a times watcher."""
    helpers = ["dashboard", "campaign_driver", "post_times", "supervise", "keep_best", "poll_status",
               "games", "mem_guard"]
    for name in helpers:
        result = probe(f"""
            import sys
            sys.path.insert(0, r"{ROOT}")
            sys.path.insert(0, r"{ROOT / "scripts"}")
            import {name}
        """)
        assert "torch" not in result["modules"], f"{name} imported torch"
        assert "stable_baselines3" not in result["modules"], f"{name} imported SB3"
        if "numpy" in result["modules"]:
            assert result["blas"] == "1", f"{name} imported numpy without capping BLAS threads"
        assert result["private"] < 200 * MB, f"{name} costs {result['private'] / MB:.0f} MB of commit"
        print(f"[ok] {name:16s} {result['private'] / MB:5.0f} MB  imports={result['modules']}")


def test_cap_blas_threads_reports_whether_it_was_in_time():
    """The ordering is the whole mechanism, so the helper says whether it won the race."""
    assert cap_blas_threads() in (True, False)
    for name in BLAS_THREAD_VARS:
        assert os.environ[name] == "1"
    import numpy  # noqa: F401

    assert cap_blas_threads() is False, "with numpy loaded the cap can no longer take effect and must say so"
    print("[ok] cap_blas_threads reports lateness")


# --------------------------------------------------------------------------- behaviour is unchanged


def test_the_in_package_monitor_matches_sb3s_monitor_exactly():
    """`EpisodeMonitor` replaced SB3's `Monitor` inside the worker. The episode record must be identical.

    Everything downstream -- `rollout/ep_rew_mean`, `ProgressCallback`, `episodes.jsonl` -- reads
    `info["episode"]`, so this is the test that says the memory fix is not a behaviour change.
    """
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.monitor import Monitor

    from ultrakill_ai.envfactory import EpisodeMonitor

    class Counter(gym.Env):
        """Three steps of known reward, then terminates, carrying two info keys."""

        observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def __init__(self):
            self.n = 0

        def reset(self, **kwargs):
            self.n = 0
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            self.n += 1
            done = self.n >= 3
            info = {"kills": self.n * 2, "deaths": 1}
            return np.zeros(2, dtype=np.float32), 1.5 * self.n, done, False, info

    keys = ("kills", "deaths")
    records = []
    for wrapper in (Monitor(Counter(), info_keywords=keys), EpisodeMonitor(Counter(), info_keywords=keys)):
        wrapper.reset()
        last = {}
        for _ in range(3):
            _, _, terminated, truncated, info = wrapper.step(0)
            if terminated or truncated:
                last = info["episode"]
        records.append(last)
    sb3, ours = records
    assert set(sb3) == set(ours), f"keys differ: {sorted(sb3)} vs {sorted(ours)}"
    assert sb3["r"] == ours["r"] == 9.0, f"return differs: {sb3['r']} vs {ours['r']}"
    assert sb3["l"] == ours["l"] == 3
    assert sb3["kills"] == ours["kills"] == 6 and sb3["deaths"] == ours["deaths"] == 1
    assert isinstance(ours["t"], float)
    print(f"[ok] EpisodeMonitor episode record identical to SB3's: {ours}")


def test_the_in_package_monitor_refuses_to_step_after_done():
    """SB3's Monitor raises here; losing that guard would hide an auto-reset bug in a worker."""
    import gymnasium as gym
    import numpy as np

    from ultrakill_ai.envfactory import EpisodeMonitor

    class OneStep(gym.Env):
        observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, **kwargs):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 0.0, True, False, {}

    monitor = EpisodeMonitor(OneStep())
    monitor.reset()
    monitor.step(0)
    try:
        monitor.step(0)
    except RuntimeError:
        print("[ok] EpisodeMonitor refuses to step after done")
    else:
        raise AssertionError("stepping a finished env must raise, as SB3's Monitor does")


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001 - a test file reports, it does not propagate
            failures += 1
            print(f"[FAIL] {test.__name__}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
