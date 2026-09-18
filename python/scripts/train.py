"""Trains a PPO agent.

    python scripts/train.py --config configs/cybergrind.yaml
    python scripts/train.py --config configs/campaign_0-1.yaml --resume models/campaign_ppo/transfer_init.zip
    python scripts/train.py --config configs/cybergrind.yaml --resume models/cybergrind/latest.zip

Parallel training with several game instances (see scripts/games.py):
    python scripts/games.py launch --count 5
    python scripts/train.py --config configs/cybergrind.yaml --num-envs 5

Watch progress with:  python scripts/dashboard.py   (or: tensorboard --logdir runs)

**THIS FILE'S MODULE BODY MUST STAY LIGHT.** `SubprocVecEnv` starts its workers with multiprocessing's
*spawn* method, and a spawn child re-executes this exact file (as `__mp_main__`, via
`runpy.run_path`) before it unpickles the env it is being asked to run. Every import at this level is
therefore paid twelve times over. It used to import torch and stable_baselines3 here, which cost
0.91 GB of committed private bytes per worker on a machine that had already run out of memory twice.
The trainer proper is `ultrakill_ai.training`, imported below only on the branch that actually runs it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__mp_main__":
    # A `SubprocVecEnv` worker, re-executing this file on its way up. It needs nothing from here, and the
    # one thing it must do is cap numpy's BLAS thread pool before anything imports numpy: OpenBLAS reserves
    # ~34 MB of committed memory per thread when it loads, which is 785 MB of the old 0.91 GB per worker.
    from ultrakill_ai.procmem import cap_blas_threads

    cap_blas_threads()
elif __name__ != "__main__":
    # Imported as a module (`import train`), which is how the tests reach these. Safe to be heavy here:
    # this branch is never taken in a worker, because a spawn child arrives as `__mp_main__` above.
    from ultrakill_ai.training import (  # noqa: F401
        ENTROPY_DIMS,
        ENTROPY_SAMPLE,
        ActionEntropyCallback,
        EntropyFloorCallback,
        EpisodeStatsCallback,
        close_vec_env,
        fill_campaign_dirs,
        fill_run_dirs,
        load_config,
        main,
    )
    from ultrakill_ai.envfactory import make_env  # noqa: F401

if __name__ == "__main__":
    from ultrakill_ai.training import main

    main()
