from __future__ import annotations

from typing import Any

import pandas as pd

from adaptive_bot.config import StrategyConfig

FEATURE_COLUMNS = (
    "atr",
    "adx",
    "center",
    "z",
    "atr_percentile",
    "atr_change",
    "ema_slope",
    "cumulative_move",
)

try:
    import cupy as cp  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    cp = None


_WILDER_KERNEL = r"""
extern "C" __global__ void wilder(const double* x, double* out, int n, int period) {
    if (blockIdx.x || threadIdx.x) return;
    int valid = 0, seed_end = -1;
    double seed = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!isnan(x[i])) {
            seed += x[i];
            if (++valid == period) { seed_end = i; break; }
        }
    }
    if (seed_end < 0) return;
    out[seed_end] = seed / period;
    for (int i = seed_end + 1; i < n; ++i) {
        if (!isnan(x[i]) && !isnan(out[i - 1]))
            out[i] = (out[i - 1] * (period - 1) + x[i]) / period;
    }
}
"""

_PERCENTILE_KERNEL = r"""
extern "C" __global__ void rolling_percentile(
    const double* x, double* out, int n, int window
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n || i < window - 1 || isnan(x[i])) return;
    int count = 0;
    for (int j = i - window + 1; j <= i; ++j) {
        if (isnan(x[j])) return;
        if (x[j] <= x[i]) ++count;
    }
    out[i] = 100.0 * count / window;
}
"""

_EMA_KERNEL = r"""
extern "C" __global__ void ema(const double* x, double* out, int n, int period) {
    if (blockIdx.x || threadIdx.x || n == 0) return;
    double alpha = 2.0 / (period + 1.0);
    double current = x[0];
    for (int i = 1; i < n; ++i) {
        current = alpha * x[i] + (1.0 - alpha) * current;
        if (i >= period - 1) out[i] = current;
    }
}
"""


def gpu_available() -> bool:
    if cp is None:
        return False
    try:
        return bool(cp.cuda.runtime.getDeviceCount())
    except cp.cuda.runtime.CUDARuntimeError:
        return False


class GpuFeatureFactory:
    """Compute the strategy feature matrix on CUDA, preserving CPU formulas."""

    def __init__(self, frame: pd.DataFrame) -> None:
        if not gpu_available():
            raise RuntimeError("CUDA research requested but CuPy cannot access a GPU")
        assert cp is not None
        self.frame = frame
        self.high = cp.asarray(frame["high"].to_numpy(dtype=float))
        self.low = cp.asarray(frame["low"].to_numpy(dtype=float))
        self.close = cp.asarray(frame["close"].to_numpy(dtype=float))
        self.volume = cp.asarray(frame["volume"].to_numpy(dtype=float))
        self._atr: dict[int, Any] = {}
        self._adx: dict[int, Any] = {}
        self._vwap: dict[int, Any] = {}
        self._ema: dict[int, Any] = {}
        self._percentile: dict[tuple[int, int], Any] = {}
        self._wilder_kernel = cp.RawKernel(_WILDER_KERNEL, "wilder")
        self._percentile_kernel = cp.RawKernel(_PERCENTILE_KERNEL, "rolling_percentile")
        self._ema_kernel = cp.RawKernel(_EMA_KERNEL, "ema")

    def build(self, config: StrategyConfig) -> pd.DataFrame:
        assert cp is not None
        atr_values = self._atr_values(config.atr_period)
        center = self._vwap_values(config.crypto_vwap_window)
        ema = self._ema_values(config.ema_period)
        z = (self.close - center) / cp.where(atr_values == 0, cp.nan, atr_values)
        atr_change = cp.concatenate((cp.asarray([cp.nan]), atr_values[1:] / atr_values[:-1] - 1))
        slope = (ema - cp.roll(ema, config.slope_lookback)) / (
            cp.where(atr_values == 0, cp.nan, atr_values) * config.slope_lookback
        )
        slope[: config.slope_lookback] = cp.nan
        cumulative = (self.close - cp.roll(self.close, 3)) / cp.where(
            atr_values == 0, cp.nan, atr_values
        )
        cumulative[:3] = cp.nan
        result = self.frame.copy()
        arrays = {
            "atr": atr_values,
            "adx": self._adx_values(config.adx_period),
            "center": center,
            "z": z,
            "atr_percentile": self._percentile_values(
                config.atr_period, config.atr_percentile_window
            ),
            "atr_change": atr_change,
            "ema_slope": slope,
            "cumulative_move": cumulative,
        }
        for name, values in arrays.items():
            result[name] = cp.asnumpy(values)
        return result

    def _true_range(self) -> Any:
        assert cp is not None
        previous = cp.concatenate((cp.asarray([cp.nan]), self.close[:-1]))
        return cp.fmax(
            self.high - self.low,
            cp.fmax(cp.abs(self.high - previous), cp.abs(self.low - previous)),
        )

    def _wilder(self, values: Any, period: int) -> Any:
        assert cp is not None
        output = cp.full(values.size, cp.nan, dtype=cp.float64)
        self._wilder_kernel((1,), (1,), (values, output, values.size, period))
        return output

    def _atr_values(self, period: int) -> Any:
        if period not in self._atr:
            self._atr[period] = self._wilder(self._true_range(), period)
        return self._atr[period]

    def _adx_values(self, period: int) -> Any:
        assert cp is not None
        if period in self._adx:
            return self._adx[period]
        up = cp.concatenate((cp.asarray([cp.nan]), cp.diff(self.high)))
        down = cp.concatenate((cp.asarray([cp.nan]), -cp.diff(self.low)))
        plus_dm = cp.where((up > down) & (up > 0), up, 0.0)
        minus_dm = cp.where((down > up) & (down > 0), down, 0.0)
        plus_dm[0] = minus_dm[0] = cp.nan
        smooth_tr = self._wilder(self._true_range(), period)
        smooth_plus = self._wilder(plus_dm, period)
        smooth_minus = self._wilder(minus_dm, period)
        plus_di = 100 * smooth_plus / cp.where(smooth_tr == 0, cp.nan, smooth_tr)
        minus_di = 100 * smooth_minus / cp.where(smooth_tr == 0, cp.nan, smooth_tr)
        denominator = plus_di + minus_di
        dx = 100 * cp.abs(plus_di - minus_di) / cp.where(denominator == 0, cp.nan, denominator)
        self._adx[period] = self._wilder(dx, period)
        return self._adx[period]

    def _vwap_values(self, window: int) -> Any:
        assert cp is not None
        if window in self._vwap:
            return self._vwap[window]
        typical = (self.high + self.low + self.close) / 3
        numerator = self._rolling_sum(typical * self.volume, window)
        denominator = self._rolling_sum(self.volume, window)
        self._vwap[window] = numerator / cp.where(denominator == 0, cp.nan, denominator)
        return self._vwap[window]

    def _rolling_sum(self, values: Any, window: int) -> Any:
        assert cp is not None
        cumulative = cp.cumsum(values)
        output = cp.full(values.size, cp.nan, dtype=cp.float64)
        output[window - 1 :] = cumulative[window - 1 :]
        if values.size > window:
            output[window:] -= cumulative[:-window]
        return output

    def _ema_values(self, period: int) -> Any:
        assert cp is not None
        if period in self._ema:
            return self._ema[period]
        output = cp.full(self.close.size, cp.nan, dtype=cp.float64)
        self._ema_kernel((1,), (1,), (self.close, output, self.close.size, period))
        self._ema[period] = output
        return output

    def _percentile_values(self, atr_period: int, window: int) -> Any:
        assert cp is not None
        key = (atr_period, window)
        if key not in self._percentile:
            values = self._atr_values(atr_period)
            output = cp.full(values.size, cp.nan, dtype=cp.float64)
            blocks = (values.size + 255) // 256
            self._percentile_kernel((blocks,), (256,), (values, output, values.size, window))
            self._percentile[key] = output
        return self._percentile[key]
