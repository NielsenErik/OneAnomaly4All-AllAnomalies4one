"""
Progress reporting for loops that take minutes to hours.

Two consumers, two formats, one call site:

  * INTERACTIVE (stderr is a tty)  -> a tqdm bar, redrawn in place.
  * REDIRECTED  (a batch, `> log`) -> a periodic one-line update with rate and
    ETA.  A tqdm bar written to a file is thousands of carriage-return lines,
    which is worse than nothing; a line every 30 s is what you actually want
    when you tail a console log at 3 a.m.

Both paths degrade to silence-free behaviour if tqdm is missing, so nothing
here can break a run.

Why this exists: a `SurvivalPC` fit on real C-MAPSS is two 50-epoch training
runs plus n_bins circuit passes per prediction, and it printed NOTHING between
start and finish.  A run that is merely slow and a run that is hung look
identical from outside, and the only way to tell them apart was to wait.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Iterable, Iterator, Optional

try:                                   # optional; never required
    from tqdm.auto import tqdm as _tqdm
except Exception:                      # pragma: no cover - depends on env
    _tqdm = None

# Set TS_PROGRESS=0 to silence everything (useful inside tests).
_ENABLED = os.environ.get("TS_PROGRESS", "1") not in ("0", "false", "no")


def _fmt(sec: float) -> str:
    if sec != sec or sec in (float("inf"), -float("inf")):
        return "??:??"
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def track(iterable: Iterable, desc: str, total: Optional[int] = None,
          log: Optional[Callable[[str], None]] = None,
          every_s: float = 30.0, indent: str = "    ") -> Iterator:
    """
    Wrap `iterable` with progress reporting.

    `log` is an optional sink (e.g. `RunLogger.info`) that also receives the
    periodic lines, so the per-run log records how long each phase took even
    when the console log is elsewhere.
    """
    if not _ENABLED:
        yield from iterable
        return

    if total is None:
        try:
            total = len(iterable)                      # type: ignore[arg-type]
        except Exception:
            total = None

    if is_tty() and _tqdm is not None:
        yield from _tqdm(iterable, desc=f"{indent}{desc}", total=total,
                         leave=False, dynamic_ncols=True, file=sys.stderr)
        return

    t0 = time.time()
    last = t0
    n = 0
    for item in iterable:
        yield item
        n += 1
        now = time.time()
        if now - last < every_s:
            continue
        last = now
        rate = n / max(now - t0, 1e-9)
        if total:
            eta = (total - n) / max(rate, 1e-9)
            msg = (f"{indent}{desc}: {n}/{total} ({100.0 * n / total:.0f}%) · "
                   f"{rate:.2f} it/s · elapsed {_fmt(now - t0)} · ETA {_fmt(eta)}")
        else:
            msg = (f"{indent}{desc}: {n} · {rate:.2f} it/s · "
                   f"elapsed {_fmt(now - t0)}")
        print(msg, file=sys.stderr, flush=True)
        if log is not None:
            log(msg)

    if total and n and time.time() - t0 > every_s:
        msg = f"{indent}{desc}: done {n}/{total} in {_fmt(time.time() - t0)}"
        print(msg, file=sys.stderr, flush=True)
        if log is not None:
            log(msg)


class Phase:
    """
    Time one named phase and report it when it ends.

    For work that is one long call rather than a loop (a structure build, a
    baseline fit), so the log still shows where the time went.
    """

    def __init__(self, desc: str, log: Optional[Callable[[str], None]] = None,
                 indent: str = "    "):
        self.desc, self.log, self.indent = desc, log, indent

    def __enter__(self) -> "Phase":
        self.t0 = time.time()
        if _ENABLED and is_tty():
            print(f"{self.indent}{self.desc} ...", file=sys.stderr, flush=True)
        return self

    def __exit__(self, *exc) -> None:
        dt = time.time() - self.t0
        if not _ENABLED or dt < 1.0:
            return
        msg = f"{self.indent}{self.desc}: {_fmt(dt)}"
        print(msg, file=sys.stderr, flush=True)
        if self.log is not None:
            self.log(msg)
