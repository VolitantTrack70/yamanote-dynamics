"""Measuring station dwell time from ODPT real-time train positions.

Dwell is the one quantity this project needs and cannot derive. Circuit time =
run time + margin + dwell is one equation in two unknowns, so Phase 3 has to
infer dwell (39-50 s) from documented margin practice rather than measure it.
This module is the path to measuring it directly.

**How it works, and why it works.** The `odpt:Train` feed reports each train's
`odpt:fromStation` and `odpt:toStation`. The key property -- verified against the
live feed, not assumed -- is that **`odpt:toStation` is null exactly when a train
is standing at a platform**, with `fromStation` naming that platform. Once it
departs, `toStation` is populated with the next station. So a dwell is simply a
contiguous run of samples in which a given train reports no `toStation`.

**Polling makes every measurement interval-censored, and this module keeps it
that way.** If a train is first seen standing at t1 and last seen standing at t2,
the true dwell is at least t2 - t1, but no more than the gap between the last
sample that showed it running and the first sample that shows it running again.
Reporting the midpoint as though it were a measurement would overstate precision.
Both bounds are carried through to the aggregate.

**Access.** JR East real-time data requires a registered consumer key. The
unauthenticated mirror at `api-public.odpt.org` carries only a subset of
operators -- Toei at the time of writing, and no JR East at all -- so the
Yamanote cannot be measured without a key. The public endpoint is still useful:
it exercises this entire pipeline against real live data from another operator,
which is a far better test than synthetic fixtures.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PUBLIC_ENDPOINT = "https://api-public.odpt.org/api/v4"
KEYED_ENDPOINT = "https://api.odpt.org/api/v4"

YAMANOTE = "odpt.Railway:JR-East.Yamanote"

#: Environment variable holding the ODPT consumer key, if one is available.
KEY_ENV = "ODPT_CONSUMER_KEY"


class OdptError(RuntimeError):
    pass


@dataclass
class OdptClient:
    """Minimal ODPT client.

    Falls back to the unauthenticated mirror when no key is configured, so the
    pipeline is runnable either way -- just against different operators.
    """

    consumer_key: str | None = None
    timeout_s: float = 45.0

    @classmethod
    def from_env(cls) -> "OdptClient":
        return cls(consumer_key=os.environ.get(KEY_ENV) or None)

    @property
    def base(self) -> str:
        return KEYED_ENDPOINT if self.consumer_key else PUBLIC_ENDPOINT

    @property
    def authenticated(self) -> bool:
        return bool(self.consumer_key)

    def trains(self, railway: str | None = None) -> list[dict]:
        """Current train positions, optionally filtered to one railway."""
        params: dict[str, str] = {}
        if railway:
            params["odpt:railway"] = railway
        if self.consumer_key:
            params["acl:consumerKey"] = self.consumer_key

        url = f"{self.base}/odpt:Train"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        req = urllib.request.Request(url, headers={"User-Agent": "tokyoline-model/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read()[:200].decode("utf-8", errors="replace")
            if exc.code == 403:
                raise OdptError(
                    f"HTTP 403 from {self.base}: {body.strip()}. "
                    f"Set {KEY_ENV} to a registered ODPT consumer key. JR East "
                    "real-time data is not on the unauthenticated mirror."
                ) from exc
            raise OdptError(f"HTTP {exc.code} from {url}: {body}") from exc


def short_station(uri: str | None) -> str | None:
    """'odpt.Station:JR-East.Yamanote.Shibuya' -> 'Shibuya'."""
    if not uri:
        return None
    return uri.rsplit(".", 1)[-1]


@dataclass
class Sample:
    """One observation of one train at one moment."""

    t: float
    train: str
    railway: str
    from_station: str | None
    to_station: str | None
    delay_s: int | None

    @property
    def standing(self) -> bool:
        """True when the train is at a platform rather than between stations."""
        return self.to_station is None


@dataclass
class DwellEpisode:
    """One observed stop, with the censoring interval kept explicit."""

    train: str
    railway: str
    station: str
    first_standing_t: float
    last_standing_t: float
    prev_running_t: float | None
    next_running_t: float | None

    @property
    def lower_bound_s(self) -> float:
        """Shortest dwell consistent with the observations."""
        return self.last_standing_t - self.first_standing_t

    @property
    def upper_bound_s(self) -> float | None:
        """Longest dwell consistent with the observations.

        None when the episode is not bracketed by running observations on both
        sides -- an episode that was already in progress when sampling started,
        or still in progress when it stopped, is censored open-ended and must
        not be counted.
        """
        if self.prev_running_t is None or self.next_running_t is None:
            return None
        return self.next_running_t - self.prev_running_t

    @property
    def bracketed(self) -> bool:
        return self.upper_bound_s is not None

    @property
    def midpoint_s(self) -> float | None:
        if not self.bracketed:
            return None
        return 0.5 * (self.lower_bound_s + self.upper_bound_s)


def episodes_from_samples(samples: list[Sample]) -> list[DwellEpisode]:
    """Group a time-ordered sample stream into per-train standing episodes."""
    by_train: dict[str, list[Sample]] = {}
    for s in sorted(samples, key=lambda x: x.t):
        by_train.setdefault(s.train, []).append(s)

    episodes: list[DwellEpisode] = []
    for train, series in by_train.items():
        i = 0
        while i < len(series):
            if not series[i].standing:
                i += 1
                continue

            station = series[i].from_station
            j = i
            while (j + 1 < len(series) and series[j + 1].standing
                   and series[j + 1].from_station == station):
                j += 1

            prev_running = None
            for k in range(i - 1, -1, -1):
                if not series[k].standing:
                    prev_running = series[k].t
                    break
            next_running = None
            for k in range(j + 1, len(series)):
                if not series[k].standing:
                    next_running = series[k].t
                    break

            if station:
                episodes.append(DwellEpisode(
                    train=train, railway=series[i].railway, station=station,
                    first_standing_t=series[i].t, last_standing_t=series[j].t,
                    prev_running_t=prev_running, next_running_t=next_running,
                ))
            i = j + 1
    return episodes


@dataclass
class DwellStats:
    """Aggregate dwell across many episodes, as an interval."""

    n_episodes: int
    n_bracketed: int
    mean_lower_s: float
    mean_upper_s: float
    per_station: dict[str, tuple[int, float, float]] = field(default_factory=dict)

    @property
    def mean_estimate_s(self) -> float:
        return 0.5 * (self.mean_lower_s + self.mean_upper_s)

    def summary(self) -> str:
        return (f"{self.n_bracketed} bracketed episodes: mean dwell in "
                f"[{self.mean_lower_s:.1f}, {self.mean_upper_s:.1f}] s "
                f"(midpoint {self.mean_estimate_s:.1f} s)")


def aggregate(episodes: list[DwellEpisode]) -> DwellStats | None:
    """Mean dwell bounds over all fully bracketed episodes."""
    usable = [e for e in episodes if e.bracketed]
    if not usable:
        return None

    lower = sum(e.lower_bound_s for e in usable) / len(usable)
    upper = sum(e.upper_bound_s for e in usable) / len(usable)

    per_station: dict[str, list[DwellEpisode]] = {}
    for e in usable:
        per_station.setdefault(e.station, []).append(e)

    return DwellStats(
        n_episodes=len(episodes),
        n_bracketed=len(usable),
        mean_lower_s=lower,
        mean_upper_s=upper,
        per_station={
            st: (len(eps),
                 sum(e.lower_bound_s for e in eps) / len(eps),
                 sum(e.upper_bound_s for e in eps) / len(eps))
            for st, eps in sorted(per_station.items())
        },
    )


def collect(client: OdptClient, *, railway: str | None, duration_s: float,
            interval_s: float = 10.0, verbose: bool = True) -> list[Sample]:
    """Poll the feed for a while and return every observation.

    `interval_s` sets the censoring width: a dwell can only be pinned to within
    roughly one polling interval on each side, so 10 s sampling bounds a 40 s
    dwell to about [30, 50] s. Averaging many episodes narrows the mean but not
    the individual measurements.
    """
    samples: list[Sample] = []
    deadline = time.time() + duration_s
    polls = 0

    while time.time() < deadline:
        t = time.time()
        try:
            trains = client.trains(railway=railway)
        except OdptError as exc:
            if verbose:
                print(f"  poll failed: {exc}")
            break

        polls += 1
        for d in trains:
            samples.append(Sample(
                t=t,
                train=d.get("owl:sameAs") or d.get("odpt:trainNumber", "?"),
                railway=d.get("odpt:railway", "?"),
                from_station=short_station(d.get("odpt:fromStation")),
                to_station=short_station(d.get("odpt:toStation")),
                delay_s=d.get("odpt:delay"),
            ))

        if verbose:
            standing = sum(1 for d in trains if not d.get("odpt:toStation"))
            print(f"  poll {polls:3d}  {len(trains):3d} trains, "
                  f"{standing:3d} standing", flush=True)

        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(interval_s, remaining))

    return samples


def save_samples(samples: list[Sample], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([{
            "t": s.t, "train": s.train, "railway": s.railway,
            "from": s.from_station, "to": s.to_station, "delay": s.delay_s,
        } for s in samples], fh, ensure_ascii=False, indent=1)


def load_samples(path: Path | str) -> list[Sample]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [Sample(t=d["t"], train=d["train"], railway=d["railway"],
                   from_station=d["from"], to_station=d["to"],
                   delay_s=d.get("delay")) for d in raw]
