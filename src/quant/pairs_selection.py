"""
pairs_selection.py — Phase 1: walk-forward pair selection + point-in-time spread features.
Produces DataFrames shaped for the `pairs` and `pair_metrics` tables.

Classes: PairScreenConfig, StatArbPairScreener (SSD -> EG w/ FDR control -> I(1) legs
-> |beta| bounds -> half-life -> Hurst), SpreadFeatureEngine (trailing beta/spread/
z-score/vol/skew/slope + live breakdown monitor), WalkForwardOrchestrator.

No-lookahead guarantees
------------------------
1. Screener only ever sees `prices.iloc[t0-F+1 : t0+1]` — trading data is
   structurally unreachable during selection.
2. Trading window starts strictly AFTER t0 (no shared bar).
3. Every trading-window value is a trailing statistic (bars <= t only).
4. No survivorship requirement: a delisted leg just stops producing rows.
5. `half_life`/`adf_pvalue` in pair_metrics are FORMATION values (known at t0),
   held constant through the window; `rolling_half_life`/`pair_active` are
   trailing and re-evaluated daily (see SpreadFeatureEngine).

Two defaults beyond the base spec (not opt-in)
------------------------------------------------
* FDR control (Benjamini-Hochberg, config.fdr_q) on the EG step instead of a flat
  p<=0.05 cutoff — a flat cutoff lets ~5% of non-cointegrated pairs through every
  window by chance, however large the candidate pool.
* Live breakdown monitor: `pair_active` (sticky within a window) flips False once
  the rolling half-life or z-score persistence breaks down mid-trading-window.
  Requires 3 new pair_metrics columns:
    ALTER TABLE pair_metrics ADD COLUMN rolling_half_life DOUBLE PRECISION;
    ALTER TABLE pair_metrics ADD COLUMN pair_active BOOLEAN NOT NULL DEFAULT TRUE;
    ALTER TABLE pair_metrics ADD COLUMN coint_qvalue DOUBLE PRECISION;

Conventions: a row dated t = "known at the close of t" (a backtester acts on it at
t+1+). Pair identity = (window_id, stock_a, stock_b) until `pairs` is upserted and
pair_id comes back.

DB I/O (reading prices, writing pairs/pair_metrics) lives in market_data_io.py —
this module is a pure pandas/numpy transform with no side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.stattools import adfuller, coint

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PairScreenConfig:
    # ── Walk-forward geometry (trading days) ──
    formation_window: int = 252          # F
    trading_window: int = 63             # T

    # ── Universe / column names ──
    symbol_col: str = "Symbol"
    sector_col: str = "GICS Sector"
    use_log_prices: bool = True          # regress / build spread on log prices

    # ── Step 1: SSD distance screen ──
    ssd_top_n: int = 20
    ssd_scope: str = "sector"            # "sector" | "universe"
    min_correlation: Optional[float] = None   # optional formation-return corr floor

    # ── Step 2: Engle-Granger, controlled for multiple testing ──
    # Screening many SSD candidates per window means many EG tests at once; a flat
    # p <= 0.05 cutoff lets a fixed 5% of them through by chance. Benjamini-Hochberg
    # (FDR control) is therefore the PRIMARY gate. adf_pvalue_max is only a loose
    # backstop cap so a generous fdr_q can never wave through a clearly weak p-value.
    fdr_q: float = 0.05                  # target false discovery rate for the EG step
    adf_pvalue_max: float = 0.10         # backstop cap on the raw (unadjusted) p-value
    adf_maxlag: Optional[int] = None     # None -> statsmodels default
    use_mackinnon_pvalue: bool = False   # True: FDR-control coint() p-value instead (see note)

    # ── Degeneracy guards (EG is only meaningful if both legs are I(1)) ──
    require_i1_legs: bool = True         # each leg's own ADF p must be > leg_adf_pvalue_min
    leg_adf_pvalue_min: float = 0.05
    hedge_ratio_abs_min: Optional[float] = 0.2   # bounds on |formation beta|; None disables
    hedge_ratio_abs_max: Optional[float] = 5.0

    # ── Step 3: OU half-life (exclusive lower, inclusive upper) ──
    hl_min: float = 1.0
    hl_max: float = 30.0

    # ── Step 4: Hurst ──
    hurst_max: float = 0.5
    hurst_max_lag: int = 20

    # ── Trading-window features (all trailing) ──
    spread_mode: str = "frozen"          # "frozen" | "rolling"
    beta_window: int = 60
    zscore_window: int = 20
    vol_short: int = 20
    vol_long: int = 60
    skew_window: int = 60
    slope_window: int = 5
    vol_on_changes: bool = True

    # ── Trading-window cointegration-breakdown monitor (on by default) ──
    # A pair selected on formation data can stop mean-reverting mid-window. This
    # tracks it live and flips `pair_active` to False (and keeps it False for the
    # rest of the window) once that happens, so a backtester can stop opening new
    # positions / flatten. It reuses hl_min/hl_max as the acceptable half-life band.
    breakdown_window: int = 60           # trailing window for the rolling OU check
    z_exit: float = 4.0                  # |z| at or above this counts as "extreme"
    max_stall_days: int = 10             # consecutive extreme days before flagging stalled

    def __post_init__(self) -> None:
        if self.ssd_scope not in ("sector", "universe"):
            raise ValueError("ssd_scope must be 'sector' or 'universe'")
        if self.spread_mode not in ("frozen", "rolling"):
            raise ValueError("spread_mode must be 'frozen' or 'rolling'")
        if self.formation_window < 30 or self.trading_window < 1:
            raise ValueError("formation_window must be >= 30 and trading_window >= 1")
        if (self.hedge_ratio_abs_min is not None and self.hedge_ratio_abs_max is not None
                and not self.hedge_ratio_abs_min < self.hedge_ratio_abs_max):
            raise ValueError("hedge_ratio_abs_min must be < hedge_ratio_abs_max")
        if not self.hl_min < self.hl_max:
            raise ValueError("hl_min must be < hl_max")
        if not 0.0 < self.fdr_q <= 1.0:
            raise ValueError("fdr_q must be in (0, 1]")
        if self.formation_window < self.warmup:
            raise ValueError(
                f"formation_window={self.formation_window} is shorter than the feature "
                f"warm-up ({self.warmup}); first trading rows would be NaN."
            )

    @property
    def warmup(self) -> int:
        """Bars needed before t0 so the first trading day has every feature."""
        longest = max(self.zscore_window, self.vol_long, self.skew_window,
                     self.slope_window, self.breakdown_window)
        return self.beta_window + longest + 1


# NOTE: adf_pvalue is the OLS-residual ADF p (matches the original spec). coint_pvalue
# is statsmodels' MacKinnon-corrected version (use_mackinnon_pvalue=True to use it for
# FDR control instead). Either way, FDR control (not a flat cutoff) decides pass/fail:
# testing ~N candidates per window at a flat p<=0.05 lets ~5% of non-cointegrated pairs
# through EVERY window by chance, so benjamini_hochberg() controls the false-discovery
# rate across the whole candidate batch instead (see StatArbPairScreener.screen).


def benjamini_hochberg(pvalues: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg step-up FDR control.
    Returns (reject: bool array, adjusted_pvalue/qvalue: float array), both aligned
    to the input order. NaN p-values are ignored (never rejected, qvalue stays NaN).
    """
    p = np.asarray(pvalues, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    qval = np.full(p.shape, np.nan)
    finite = np.where(np.isfinite(p))[0]
    m = finite.size
    if m == 0:
        return reject, qval

    order = finite[np.argsort(p[finite], kind="stable")]
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    # Adjusted p-values: smallest q at which each ranked p-value would be rejected,
    # enforced monotone via a reverse cumulative minimum (standard BH construction).
    raw_adj = ranked * m / ranks
    adj = np.minimum.accumulate(raw_adj[::-1])[::-1]
    qval[order] = np.minimum(adj, 1.0)

    passed = ranked <= (ranks / m) * q
    if passed.any():
        k = int(np.max(np.where(passed)[0]))       # largest i with p_(i) <= (i/m)*q
        reject[order[: k + 1]] = True
    return reject, qval


# ─────────────────────────────────────────────
# 2. LOW-LEVEL STATISTICS (formation-window only)
# ─────────────────────────────────────────────

def ols_fit(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Closed-form OLS y = alpha + beta*x. Returns (alpha, beta)."""
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 0.0:
        return np.nan, np.nan
    beta = float(xc @ (y - y.mean())) / denom
    return float(y.mean() - beta * x.mean()), beta


def ou_half_life(spread: np.ndarray) -> float:
    """
    Half-life from the AR(1) form s_t = c + phi*s_{t-1} + e:  HL = -ln2 / ln(phi).
    Returns +inf when phi is outside (0, 1) (no mean reversion / oscillating).
    """
    lag, delta = spread[:-1], np.diff(spread)
    _, lam = ols_fit(delta, lag)          # delta = a + lam*lag, phi = 1 + lam
    phi = 1.0 + lam
    if not np.isfinite(phi) or not 0.0 < phi < 1.0:
        return np.inf
    return float(-np.log(2.0) / np.log(phi))


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Scaling of the std of lagged differences: std(x_{t+k} - x_t) ~ k^H.
    H < 0.5 mean-reverting, ~0.5 random walk, > 0.5 trending.
    """
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(series[k:] - series[:-k]) for k in lags])
    if np.any(tau <= 0) or not np.all(np.isfinite(tau)):
        return np.nan
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


# ─────────────────────────────────────────────
# 3. STAT-ARB PAIR SCREENER  (formation window ONLY)
# ─────────────────────────────────────────────

@dataclass
class ScreenResult:
    candidates: pd.DataFrame    # every SSD-shortlisted pair with all flags
    funnel: dict                # sequential counts for diagnostics

    @property
    def valid(self) -> pd.DataFrame:
        if self.candidates.empty:
            return self.candidates
        return self.candidates[self.candidates["is_valid"]].reset_index(drop=True)


class StatArbPairScreener:
    """
    Funnel: SSD top-N -> Engle-Granger ADF -> both legs I(1) -> |beta| bounds
            -> OU half-life -> Hurst.
    `screen()` receives ONLY the formation window; it has no way to see later data.
    """

    def __init__(self, config: PairScreenConfig):
        self.cfg = config

    # ── public ───────────────────────────────
    def screen(self, formation_prices: pd.DataFrame, sector_map: pd.Series) -> ScreenResult:
        cfg = self.cfg
        tickers = [t for t in formation_prices.columns if t in sector_map.index]
        px = formation_prices[tickers]
        # Complete, positive history *inside the formation window* only.
        px = px.dropna(axis=1, how="any")
        px = px.loc[:, (px > 0).all(axis=0)]

        if cfg.ssd_scope == "universe":
            groups = {"__UNIVERSE__": sorted(px.columns)}
        else:
            groups = {}
            for t in px.columns:
                groups.setdefault(sector_map[t], []).append(t)
            groups = {s: sorted(ts) for s, ts in groups.items()}

        rows: list[dict] = []
        self._leg_cache: dict[str, float] = {}       # per-window: ticker -> leg ADF p
        n_pairs_total = 0
        for group, cols in groups.items():
            if len(cols) < 2:
                continue
            n_pairs_total += len(cols) * (len(cols) - 1) // 2
            block = px[cols]
            for rank, (i, j, ssd) in enumerate(self._top_ssd_pairs(block.to_numpy(), cfg.ssd_top_n), 1):
                a, b = cols[i], cols[j]
                sector = (
                    sector_map[a] if sector_map[a] == sector_map[b] else "CROSS_SECTOR"
                )
                rows.append(self._test_pair(block[a].to_numpy(), block[b].to_numpy(),
                                            a, b, sector, ssd, rank))

        cand = pd.DataFrame(rows)
        if not cand.empty:
            self._apply_fdr_control(cand)
        return ScreenResult(cand, self._funnel(cand, len(px.columns), n_pairs_total))

    def _apply_fdr_control(self, cand: pd.DataFrame) -> None:
        """
        In-place: BH-adjust the cointegration test across every candidate tested in
        this window (the whole `cand` set = the family of hypotheses tested together).
        Overwrites the flat-threshold `passes_cointegration` placeholder from _test_pair.
        """
        cfg = self.cfg
        p_used = cand["coint_pvalue"] if cfg.use_mackinnon_pvalue else cand["adf_pvalue"]
        reject, qval = benjamini_hochberg(p_used.to_numpy(), cfg.fdr_q)
        cand["coint_qvalue"] = qval
        cand["passes_cointegration"] = reject & (p_used.to_numpy() <= cfg.adf_pvalue_max)
        cand["is_valid"] = (cand["passes_correlation"] & cand["passes_cointegration"]
                            & cand["passes_i1_legs"] & cand["passes_hedge_bounds"]
                            & cand["passes_half_life"] & cand["passes_hurst"])

    # ── step 1: vectorised SSD on rebased prices ─────────────
    @staticmethod
    def _top_ssd_pairs(prices: np.ndarray, top_n: int) -> list[tuple[int, int, float]]:
        """
        SSD_ij = sum_t (N_i,t - N_j,t)^2 with N = P / P_0 (rebased on the FIRST
        formation bar). Uses SSD = |Ni|^2 + |Nj|^2 - 2 Ni.Nj => one matmul, no pair loop.
        """
        n = prices.shape[1]
        norm = prices / prices[0]
        gram = norm.T @ norm
        d = np.diag(gram)
        ssd = np.maximum(d[:, None] + d[None, :] - 2.0 * gram, 0.0)
        iu = np.triu_indices(n, k=1)
        vals = ssd[iu]
        k = min(top_n, vals.size)
        if k == 0:
            return []
        sel = np.argpartition(vals, k - 1)[:k]
        sel = sel[np.argsort(vals[sel], kind="stable")]
        return [(int(iu[0][s]), int(iu[1][s]), float(vals[s])) for s in sel]

    # ── steps 2-4 on one pair ─────────────────
    def _test_pair(self, pa: np.ndarray, pb: np.ndarray, a: str, b: str,
                   sector: str, ssd: float, rank: int) -> dict:
        cfg = self.cfg
        y, x = (np.log(pa), np.log(pb)) if cfg.use_log_prices else (pa, pb)

        alpha, beta = ols_fit(y, x)
        resid = y - alpha - beta * x                 # formation spread, demeaned
        spread_mean = alpha                          # mean of (y - beta*x)
        spread_std = float(resid.std(ddof=1))

        corr = float(np.corrcoef(np.diff(pa) / pa[:-1], np.diff(pb) / pb[:-1])[0, 1])

        adf_p = coint_p = np.nan
        try:
            adf_p = float(adfuller(resid, maxlag=cfg.adf_maxlag, regression="c", autolag="AIC",
                                    result_object=False)[1])
            coint_p = float(coint(y, x, trend="c", maxlag=cfg.adf_maxlag, autolag="aic")[1])
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("ADF/coint failed for %s/%s: %s", a, b, exc)

        # Precondition of EG: both legs non-stationary. Cached: a ticker recurs in many pairs.
        leg_p = {}
        for name, arr in ((a, y), (b, x)):
            if name not in self._leg_cache:
                self._leg_cache[name] = self._leg_adf_p(arr)
            leg_p[name] = self._leg_cache[name]

        hl = ou_half_life(resid)
        hurst = hurst_exponent(resid, cfg.hurst_max_lag)

        passes_corr = cfg.min_correlation is None or corr >= cfg.min_correlation
        # passes_cointegration/is_valid below are PLACEHOLDERS: the real, FDR-controlled
        # decision is made once across the whole window in _apply_fdr_control, since BH
        # needs the full set of p-values tested together, not one pair at a time.
        passes_i1 = (not cfg.require_i1_legs) or bool(
            np.isfinite(leg_p[a]) and np.isfinite(leg_p[b])
            and leg_p[a] > cfg.leg_adf_pvalue_min and leg_p[b] > cfg.leg_adf_pvalue_min)
        lo, hi = cfg.hedge_ratio_abs_min, cfg.hedge_ratio_abs_max
        passes_beta = bool(np.isfinite(beta)
                           and (lo is None or abs(beta) >= lo)
                           and (hi is None or abs(beta) <= hi))
        passes_hl = bool(cfg.hl_min < hl <= cfg.hl_max)
        passes_hurst = bool(np.isfinite(hurst) and hurst < cfg.hurst_max)

        return dict(
            stock_a=a, stock_b=b, sector=sector,
            ssd=ssd, ssd_rank=rank, correlation=corr,
            hedge_ratio=beta, intercept=alpha,
            spread_mean=spread_mean, spread_std=spread_std,
            adf_pvalue=adf_p, coint_pvalue=coint_p,
            half_life=hl, hurst=hurst, formation_n=len(y),
            leg_a_adf_pvalue=leg_p[a], leg_b_adf_pvalue=leg_p[b],
            passes_correlation=passes_corr, passes_cointegration=False,   # set by BH, below
            passes_i1_legs=passes_i1, passes_hedge_bounds=passes_beta,
            passes_half_life=passes_hl, passes_hurst=passes_hurst,
            is_valid=False,                                              # set by BH, below
        )

    def _leg_adf_p(self, arr: np.ndarray) -> float:
        try:
            return float(adfuller(arr, maxlag=self.cfg.adf_maxlag, regression="c",
                                  autolag="AIC", result_object=False)[1])
        except (ValueError, np.linalg.LinAlgError):
            return np.nan

    @staticmethod
    def _funnel(cand: pd.DataFrame, n_tickers: int, n_pairs_total: int) -> dict:
        steps = [("n_coint", ["passes_correlation", "passes_cointegration"]),
                 ("n_i1_legs", ["passes_i1_legs"]),
                 ("n_hedge_bounds", ["passes_hedge_bounds"]),
                 ("n_half_life", ["passes_half_life"]),
                 ("n_hurst", ["passes_hurst"])]
        out = dict(n_tickers=n_tickers, n_pairs_total=n_pairs_total, n_ssd=len(cand),
                   **{k: 0 for k, _ in steps}, n_valid=0)
        if cand.empty:
            return out
        m = pd.Series(True, index=cand.index)
        for key, cols in steps:                      # sequential (cumulative) funnel
            for c in cols:
                m &= cand[c]
            out[key] = int(m.sum())
        out["n_valid"] = out["n_hurst"]
        return out


# ─────────────────────────────────────────────
# 4. SPREAD FEATURE ENGINE  (trailing-only)
# ─────────────────────────────────────────────

class SpreadFeatureEngine:
    """
    Vectorised across all selected pairs at once: legs are laid out as
    (time x pair) frames and pandas' C-implemented rolling ops run on every
    column together. No .apply() on windows, no per-day loops.

    Also runs a live cointegration-breakdown monitor: a pair selected on formation
    data can stop mean-reverting partway through the trading window, and nothing
    upstream re-checks that. `rolling_half_life` is a trailing OU half-life
    re-estimated each day; `pair_active` is True while it stays inside
    [hl_min, hl_max] and the z-score hasn't been stuck beyond `z_exit` for more
    than `max_stall_days`, and FALSE FOR EVERY DAY AFTER THE FIRST BREAKDOWN in a
    given trading window (sticky, via a per-pair cumulative-min) — once a pair's
    mean reversion breaks down there is no principled reason to trust the next
    day's signal more than today's, so it does not "come back" mid-window. A
    backtester should stop opening new positions, and consider flattening open
    ones, once pair_active goes False.
    """

    FEATURE_COLS = [
        "hedge_ratio", "spread", "z_score", "dz_1", "spread_vol_20",
        "spread_skew_60", "spread_slope_5", "vol_ratio",
        "rolling_half_life", "pair_active",
    ]

    def __init__(self, config: PairScreenConfig):
        self.cfg = config

    def compute_wide(self, history: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Trailing (time x pair) feature frames over the FULL history passed in — no
        trading-window mask applied. Every column at row t is a function of bars <= t
        only, so slicing this dict at any cutoff (e.g. formation vs. trading) is safe
        and gives exactly what a causal observer would have seen at that date. This is
        the shared core `compute()` and the regime classifier both build on.
        """
        cfg = self.cfg
        px = np.log(history) if cfg.use_log_prices else history
        keys = (pairs["stock_a"] + "/" + pairs["stock_b"]).tolist()
        Y = px[pairs["stock_a"].tolist()].set_axis(keys, axis=1)
        X = px[pairs["stock_b"].tolist()].set_axis(keys, axis=1)

        if cfg.spread_mode == "frozen":
            b = pd.Series(pairs["hedge_ratio"].to_numpy(), index=keys)
            a = pd.Series(pairs["intercept"].to_numpy(), index=keys)
            s0 = pd.Series(pairs["spread_std"].to_numpy(), index=keys)
            beta = pd.DataFrame(
                np.tile(b.to_numpy(), (len(px.index), 1)),
                index=px.index,
                columns=keys,
            )
            spread = Y - X.mul(b, axis=1)
            z = (spread - a) / s0
        else:
            # Trailing OLS hedge ratio over [t-W+1, t]
            W = cfg.beta_window
            beta = Y.rolling(W).cov(X) / X.rolling(W).var()
            spread = Y - beta * X

            # Rolling z-score over [t-Z+1, t]
            z_roll = spread.rolling(cfg.zscore_window)
            sd = z_roll.std().where(lambda s: s > 0)
            z = (spread - z_roll.mean()) / sd

        base = spread.diff() if cfg.vol_on_changes else spread
        vol_s = base.rolling(cfg.vol_short).std()
        vol_l = base.rolling(cfg.vol_long).std()

        # Live cointegration-breakdown monitor (trailing, vectorised)
        bw = cfg.breakdown_window
        lag, delta = spread.shift(1), spread.diff()
        phi = 1.0 + delta.rolling(bw).cov(lag) / lag.rolling(bw).var()
        phi_ok = phi.where((phi > 0) & (phi < 1))
        rolling_hl = (-np.log(2.0) / np.log(phi_ok)).where(phi_ok.notna(), np.inf)
        is_cointegrated = rolling_hl.gt(cfg.hl_min) & rolling_hl.le(cfg.hl_max)

        extreme = z.abs() >= cfg.z_exit
        stalled = self._consecutive_true(extreme) >= cfg.max_stall_days
        active_today = is_cointegrated & ~stalled

        return {
            "hedge_ratio": beta,
            "spread": spread,
            "z_score": z,
            "dz_1": z.diff(1),
            "spread_vol_20": vol_s,
            "spread_skew_60": spread.rolling(cfg.skew_window).skew(),
            "spread_slope_5": self._rolling_slope(spread, cfg.slope_window),
            "vol_ratio": vol_s / vol_l.where(vol_l > 0),
            "rolling_half_life": rolling_hl.replace(np.inf, np.nan),
            "_active_today": active_today,   # internal: consumed by compute() below
        }

    def compute(self, history: pd.DataFrame, pairs: pd.DataFrame, t0: pd.Timestamp,
                window_id: int) -> pd.DataFrame:
        """
        history : prices for [formation_start .. trading_end] (bars <= t are the
                  only ones any output row at time t depends on).
        pairs   : valid formation pairs for this window.
        Returns long-format rows for trading dates strictly after t0.
        """
        feats = self.compute_wide(history, pairs)
        active_today = feats.pop("_active_today")

        mask = history.index > t0                       # trading dates only
        # Sticky flag, but the cummin must start AT t0, not at the start of formation:
        # rolling windows are still filling during warm-up, so active_today is noisy/False
        # there by construction, and cummin over the full frame would kill the flag before
        # trading even begins. Restrict the cumulative min to the trading slice only.
        pair_active = active_today.copy()
        pair_active.loc[mask] = active_today.loc[mask].cummin()
        feats["pair_active"] = pair_active
        idx = history.index[mask]
        n_t, n_p = len(idx), len(pairs)
        if n_t == 0:
            return pd.DataFrame()

        long = pd.DataFrame(
            {name: f.loc[mask].to_numpy().ravel(order="F") for name, f in feats.items()}
        )
        long.insert(0, "time", np.tile(idx.to_numpy(), n_p))
        pk = pairs[["stock_a", "stock_b"]].reset_index(drop=True)
        long["stock_a"] = np.repeat(pk["stock_a"].to_numpy(), n_t)
        long["stock_b"] = np.repeat(pk["stock_b"].to_numpy(), n_t)
        long["window_id"] = window_id

        # Static formation values (known at t0) -> constant over the window
        long["half_life"] = np.repeat(pairs["half_life"].to_numpy(), n_t)
        long["adf_pvalue"] = np.repeat(pairs["adf_pvalue"].to_numpy(), n_t)
        long["coint_qvalue"] = np.repeat(pairs["coint_qvalue"].to_numpy(), n_t)
        long["lookback_window"] = self.cfg.beta_window

        long = long.replace([np.inf, -np.inf], np.nan)
        # NOT NULL columns in pair_metrics
        long = long.dropna(subset=["hedge_ratio", "spread", "z_score"])
        return long.reset_index(drop=True)

    @staticmethod
    def _consecutive_true(mask: pd.DataFrame) -> pd.DataFrame:
        """
        Per column: length of the run of True values ending at each row (0 if the
        row itself is False). Vectorised: a cumulative count of True, minus that
        same running count as it last stood at the most recent False.
        """
        cum = mask.cumsum()
        cum_at_last_false = cum.where(~mask).ffill().fillna(0)
        return cum - cum_at_last_false

    @staticmethod
    def _rolling_slope(df: pd.DataFrame, k: int) -> pd.DataFrame:
        """OLS slope of the last k points vs. 0..k-1, via one strided dot product."""
        arr = df.to_numpy(dtype=float)
        if arr.shape[0] < k:
            return pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        t = np.arange(k, dtype=float)
        w = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
        win = sliding_window_view(arr, k, axis=0)          # (T-k+1, P, k)
        out = np.full_like(arr, np.nan)
        out[k - 1:] = win @ w
        return pd.DataFrame(out, index=df.index, columns=df.columns)


# ─────────────────────────────────────────────
# 5. WALK-FORWARD ORCHESTRATOR
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    formation_start: pd.Timestamp
    t0: pd.Timestamp                     # last formation bar
    trading_start: pd.Timestamp          # first bar strictly after t0
    trading_end: pd.Timestamp
    pos_start: int                       # positional bounds in the price index
    pos_t0: int
    pos_end: int


@dataclass
class WalkForwardResult:
    pairs: pd.DataFrame                  # -> `pairs` table (+ window/formation columns)
    pair_metrics: pd.DataFrame           # -> `pair_metrics` table (pair_id attached later)
    funnel: pd.DataFrame                 # per-window diagnostics
    all_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)


class WalkForwardOrchestrator:
    """
    For each window:  screen on Formation [t0-F+1, t0]  ->  compute trailing
    features on Trading (t0, t0+T]  ->  next window starts at t0 + T.
    """

    def __init__(self, config: PairScreenConfig,
                 screener: Optional[StatArbPairScreener] = None,
                 engine: Optional[SpreadFeatureEngine] = None):
        self.cfg = config
        self.screener = screener or StatArbPairScreener(config)
        self.engine = engine or SpreadFeatureEngine(config)

    # ── window generation ────────────────────
    def iter_windows(self, index: pd.DatetimeIndex) -> Iterator[WalkForwardWindow]:
        F, T, n = self.cfg.formation_window, self.cfg.trading_window, len(index)
        pos_t0, wid = F - 1, 0
        while pos_t0 < n - 1:
            pos_end = min(pos_t0 + T, n - 1)          # last window may be partial
            yield WalkForwardWindow(
                window_id=wid,
                formation_start=index[pos_t0 - F + 1], t0=index[pos_t0],
                trading_start=index[pos_t0 + 1], trading_end=index[pos_end],
                pos_start=pos_t0 - F + 1, pos_t0=pos_t0, pos_end=pos_end,
            )
            pos_t0 += T
            wid += 1

    # ── main entry ───────────────────────────
    def run(self, prices: pd.DataFrame, sectors_df: pd.DataFrame) -> WalkForwardResult:
        self._validate_prices(prices)
        cfg = self.cfg
        sector_map = (sectors_df.drop_duplicates(cfg.symbol_col)
                      .set_index(cfg.symbol_col)[cfg.sector_col])

        pair_frames, metric_frames, funnels, cand_frames = [], [], [], []
        for w in self.iter_windows(prices.index):
            formation = prices.iloc[w.pos_start: w.pos_t0 + 1]
            assert formation.index[-1] == w.t0 and formation.index[-1] < w.trading_start

            res = self.screener.screen(formation, sector_map)   # formation data only
            funnels.append({"window_id": w.window_id, "t0": w.t0, **res.funnel})
            if not res.candidates.empty:
                cand_frames.append(res.candidates.assign(window_id=w.window_id, t0=w.t0))

            valid = res.valid
            logger.info("window %d  t0=%s  valid=%d", w.window_id, w.t0.date(), len(valid))
            if valid.empty:
                continue

            history = prices.iloc[w.pos_start: w.pos_end + 1]
            metrics = self.engine.compute(history, valid, w.t0, w.window_id)
            if metrics.empty:
                continue

            pair_frames.append(valid.assign(
                window_id=w.window_id, formation_start=w.formation_start, formation_end=w.t0,
                trading_start=w.trading_start, trading_end=w.trading_end))
            metric_frames.append(metrics)

        return WalkForwardResult(
            pairs=self._concat(pair_frames),
            pair_metrics=self._finalize_metrics(self._concat(metric_frames)),
            funnel=pd.DataFrame(funnels),
            all_candidates=self._concat(cand_frames),
        )

    # ── helpers ──────────────────────────────
    @staticmethod
    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _finalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df["half_life"] = df["half_life"].where(np.isfinite(df["half_life"]))   # inf -> NULL
        df["pair_active"] = df["pair_active"].astype(bool)
        df["time"] = pd.to_datetime(df["time"]).dt.date                        # DATE column
        lead = ["window_id", "stock_a", "stock_b", "time"]
        cols = lead + [c for c in SpreadFeatureEngine.FEATURE_COLS]
        cols += ["half_life", "adf_pvalue", "coint_qvalue", "lookback_window"]
        return df[cols]

    @staticmethod
    def _validate_prices(prices: pd.DataFrame) -> None:
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must be indexed by a DatetimeIndex")
        if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
            raise ValueError("prices index must be sorted ascending with no duplicates")
        if prices.columns.has_duplicates:
            raise ValueError("duplicate ticker columns")