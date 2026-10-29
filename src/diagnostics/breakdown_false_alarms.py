from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from quant.pairs_selection import PairScreenConfig


def _simulate_false_alarm_share(true_half_life: float, cfg: PairScreenConfig, sims: int = 500, burn: int = 120) -> tuple[float, float]:
    phi = math.exp(-math.log(2.0) / true_half_life)
    length = burn + cfg.trading_window
    false_any = 0
    false_final = 0

    rng = np.random.default_rng(2024 + int(true_half_life))
    for _ in range(sims):
        series = np.empty(length, dtype=float)
        noise = rng.normal(size=length)
        series[0] = noise[0]
        for i in range(1, length):
            series[i] = phi * series[i - 1] + noise[i]

        s = pd.Series(series)
        lag = s.shift(1)
        delta = s.diff()
        phi_hat = 1.0 + delta.rolling(cfg.breakdown_window).cov(lag) / lag.rolling(cfg.breakdown_window).var()
        phi_ok = phi_hat.where((phi_hat > 0) & (phi_hat < 1))
        rolling_hl = (-math.log(2.0) / np.log(phi_ok)).where(phi_ok.notna(), np.inf)
        active_today = (rolling_hl > cfg.hl_min) & (rolling_hl <= cfg.hl_max)

        trading = s.index[-cfg.trading_window :]
        active = active_today.copy()
        active.loc[trading] = active_today.loc[trading].cummin()

        if not active.loc[trading].all():
            false_any += 1
        if not active.iloc[-1]:
            false_final += 1

    return false_any / sims, false_final / sims


def _run_simulation() -> pd.DataFrame:
    cfg = PairScreenConfig()
    rows = []
    for hl_true in (5, 10, 15, 20):
        share_any, share_final = _simulate_false_alarm_share(float(hl_true), cfg)
        rows.append({
            "true_half_life": int(hl_true),
            "share_false_any_day": share_any,
            "share_false_final_day": share_final,
        })
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _run_simulation()
    df.to_csv(out_dir / "breakdown_false_alarms.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
