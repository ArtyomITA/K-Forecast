"""ARIMA/SARIMA wrapper for EBS daily row counts.

This module exposes :func:`fit_predict_arima_ebs` which expects a long dataframe
with columns ``item_id``, ``timestamp`` and ``target`` (daily counts). It fits a
seasonal ARIMA model per series, produces probabilistic forecasts (p10/p50/p90)
for the requested horizon, and collects diagnostic information such as
identified orders, AIC and fallback usage.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

LOGGER = logging.getLogger("k_forecast.arima")

try:  # Optional dependency for auto order discovery
    import pmdarima as pm  # type: ignore

    PMDARIMA_AVAILABLE = True
except Exception:  # pragma: no cover - we want graceful degradation in prod
    pm = None
    PMDARIMA_AVAILABLE = False


@dataclass
class _SeriesResult:
    item_id: str
    forecast: pd.DataFrame
    info: Dict[str, Any]


def _month_end_windows(index: pd.DatetimeIndex, days: int = 3) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if index.empty:
        return []
    periods = index.to_period("M")
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for period in periods.unique():
        month_idx = np.flatnonzero(periods == period)
        if month_idx.size == 0:
            continue
        tail_size = min(days, month_idx.size)
        window_idx = month_idx[-tail_size:]
        start = index[window_idx[0]]
        end = index[window_idx[-1]]
        windows.append((pd.Timestamp(start), pd.Timestamp(end)))
    return windows


def _detect_payroll_dates(series: pd.Series) -> List[pd.Timestamp]:
    if series.empty:
        return []
    periods = pd.PeriodIndex(series.index, freq="M")
    dates: List[pd.Timestamp] = []
    for period in periods.unique():
        mask = periods == period
        month_series = series[mask]
        if month_series.empty:
            continue
        idxmax = month_series.idxmax()
        if isinstance(idxmax, pd.Timestamp):
            dates.append(idxmax)
        else:
            dates.append(pd.Timestamp(idxmax))
    return sorted(dates)


def _validate_input(df_long: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"item_id", "timestamp", "target"}
    missing = required_cols - set(df_long.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df_long.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df.sort_values(["item_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _seasonal_naive(values: np.ndarray, seasonal_period: int, horizon: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(horizon)
    if seasonal_period <= 0:
        seasonal_period = 1
    if values.size >= seasonal_period:
        pattern = values[-seasonal_period:]
    else:
        repeats = seasonal_period // values.size + 1
        pattern = np.tile(values, repeats)[:seasonal_period]
    return np.tile(pattern, horizon // seasonal_period + 1)[:horizon]


def _manual_candidates(
    seasonal_period: int,
    max_p: int,
    max_d: int,
    max_q: int,
    max_P: int,
    max_D: int,
    max_Q: int,
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]]:
    candidates = [
        ((1, 1, 1), (0, 1, 1, seasonal_period)),
        ((2, 1, 1), (0, 1, 1, seasonal_period)),
        ((1, 1, 0), (0, 1, 1, seasonal_period)),
        ((1, 1, 1), (1, 1, 0, seasonal_period)),
        ((0, 1, 1), (0, 1, 1, seasonal_period)),
    ]

    def _within_limits(order: Tuple[int, int, int], seasonal: Tuple[int, int, int, int]) -> bool:
        p, d, q = order
        P, D, Q, _ = seasonal
        return (
            p <= max_p
            and d <= max_d
            and q <= max_q
            and P <= max_P
            and D <= max_D
            and Q <= max_Q
        )

    filtered = [cand for cand in candidates if _within_limits(*cand)]
    if not filtered:
        filtered.append(((1, min(1, max_d), 1), (0, min(1, max_D), 1, seasonal_period)))
    return filtered


def _fit_sarimax(
    y: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> SARIMAXResultsWrapper:
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        initialization="approximate_diffuse",
    )
    return model.fit(disp=False)


def fit_predict_arima_ebs(
    df_long: pd.DataFrame,
    prediction_length: int,
    freq: str = "D",
    seasonal_period: int = 7,
    conf_level: float = 0.80,
    auto_order: bool = True,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    max_P: int = 1,
    max_D: int = 1,
    max_Q: int = 1,
    min_train_points: int = 56,
    enforce_nonneg: bool = True,
    round_to_int: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Forecast EBS daily counts per bank using (S)ARIMA or seasonal naive fallback."""

    if prediction_length <= 0:
        raise ValueError("prediction_length must be > 0")

    df = _validate_input(df_long)
    total_start = time.perf_counter()

    results: List[_SeriesResult] = []
    fallback_count = 0
    metrics_mae: Dict[str, float] = {}

    freq_offset = pd.tseries.frequencies.to_offset(freq)

    for raw_item_id, group in df.groupby("item_id", sort=False):
        item_id = str(raw_item_id)
        series_start = time.perf_counter()
        group = group.copy()
        group.set_index("timestamp", inplace=True)

        full_index = pd.date_range(group.index.min(), group.index.max(), freq=freq_offset)
        target = group["target"].reindex(full_index).fillna(0.0)
        target.name = "target"

        if "is_holdout" in group.columns:
            holdout_mask = (
                group["is_holdout"].astype(bool).reindex(full_index).fillna(False)
            )
        else:
            holdout_mask = pd.Series(False, index=full_index)

        train_target = target[~holdout_mask]
        holdout_target = target[holdout_mask] if holdout_mask.any() else None

        fc_mean_vals: Optional[np.ndarray] = None
        lower_vals: Optional[np.ndarray] = None
        upper_vals: Optional[np.ndarray] = None

        if train_target.empty:
            LOGGER.warning("[ARIMA][%s] no training data after holdout filtering; fallback", item_id)
            fallback = True
        else:
            fallback = len(train_target) < min_train_points

        month_end_windows = _month_end_windows(full_index)
        payroll_dates = _detect_payroll_dates(target)

        series_info: Dict[str, Any] = {
            "item_id": item_id,
            "train_points": int(len(train_target)),
            "holdout_points": int(len(holdout_target) if holdout_target is not None else 0),
            "month_end_windows": [
                {"start": window[0].strftime("%Y-%m-%d"), "end": window[1].strftime("%Y-%m-%d")}
                for window in month_end_windows
            ],
            "payroll_dates": [dt.strftime("%Y-%m-%d") for dt in payroll_dates],
        }

        if fallback:
            fallback_count += 1
            LOGGER.info(
                "[ARIMA][%s] fallback=seasonal_naive (len=%d < %d)",
                item_id,
                len(train_target),
                min_train_points,
            )
            fc_mean_vals = _seasonal_naive(train_target.to_numpy(), seasonal_period, prediction_length)
            if holdout_target is not None and not holdout_target.empty:
                holdout_pred = _seasonal_naive(train_target.to_numpy(), seasonal_period, len(holdout_target))
                mae = float(np.mean(np.abs(holdout_pred - holdout_target.to_numpy())))
                metrics_mae[item_id] = mae
                series_info["mae_holdout"] = mae
        else:
            order: Optional[Tuple[int, int, int]] = None
            seasonal_order: Optional[Tuple[int, int, int, int]] = None
            model_result: Optional[SARIMAXResultsWrapper] = None
            best_aic = float("inf")

            if auto_order and PMDARIMA_AVAILABLE and pm is not None:
                try:
                    auto_model = pm.auto_arima(
                        train_target,
                        seasonal=True,
                        m=seasonal_period,
                        max_p=max_p,
                        max_q=max_q,
                        max_P=max_P,
                        max_Q=max_Q,
                        max_d=max_d,
                        max_D=max_D,
                        error_action="ignore",
                        suppress_warnings=True,
                        stepwise=True,
                        seasonal_test="ocsb",
                    )
                    order = auto_model.order
                    s_order = auto_model.seasonal_order
                    seasonal_order = (s_order[0], s_order[1], s_order[2], seasonal_period)
                except Exception as exc:  # pragma: no cover - defensive path
                    LOGGER.warning(
                        "[ARIMA][%s] auto_arima failed (%s); switching to manual grid",
                        item_id,
                        exc,
                    )

            if order is None or seasonal_order is None:
                for cand_order, cand_seasonal in _manual_candidates(
                    seasonal_period,
                    max_p,
                    max_d,
                    max_q,
                    max_P,
                    max_D,
                    max_Q,
                ):
                    try:
                        candidate_result = _fit_sarimax(train_target, cand_order, cand_seasonal)
                    except (ValueError, LinAlgError) as exc:
                        LOGGER.debug(
                            "[ARIMA][%s] candidate order=%s seasonal=%s failed (%s)",
                            item_id,
                            cand_order,
                            cand_seasonal,
                            exc,
                        )
                        continue
                    if candidate_result.aic < best_aic:
                        best_aic = candidate_result.aic
                        order = cand_order
                        seasonal_order = cand_seasonal
                        model_result = candidate_result

            if order is None or seasonal_order is None or model_result is None:
                LOGGER.warning(
                    "[ARIMA][%s] no valid SARIMA model found; using seasonal naive fallback",
                    item_id,
                )
                fallback_count += 1
                fc_mean_vals = _seasonal_naive(train_target.to_numpy(), seasonal_period, prediction_length)
                if holdout_target is not None and not holdout_target.empty:
                    holdout_pred = _seasonal_naive(train_target.to_numpy(), seasonal_period, len(holdout_target))
                    mae = float(np.mean(np.abs(holdout_pred - holdout_target.to_numpy())))
                    metrics_mae[item_id] = mae
                    series_info["mae_holdout"] = mae
            else:
                if model_result is None:
                    model_result = _fit_sarimax(train_target, order, seasonal_order)
                    best_aic = model_result.aic

                if holdout_target is not None and not holdout_target.empty:
                    holdout_forecast = model_result.get_forecast(len(holdout_target))
                    holdout_mean = holdout_forecast.predicted_mean
                    mae = float(np.mean(np.abs(holdout_mean.to_numpy() - holdout_target.to_numpy())))
                    metrics_mae[item_id] = mae
                    series_info["mae_holdout"] = mae

                try:
                    if holdout_target is not None and not holdout_target.empty:
                        final_series = target
                    else:
                        final_series = train_target
                    final_model = _fit_sarimax(final_series, order, seasonal_order)
                except (ValueError, LinAlgError) as exc:  # pragma: no cover
                    LOGGER.warning(
                        "[ARIMA][%s] final fit failed (%s); reverting to training fit",
                        item_id,
                        exc,
                    )
                    final_model = model_result

                forecast_res = final_model.get_forecast(prediction_length)
                fc_mean_vals = forecast_res.predicted_mean.to_numpy()
                conf_int = forecast_res.conf_int(alpha=1 - conf_level)
                lower_vals = conf_int.iloc[:, 0].to_numpy()
                upper_vals = conf_int.iloc[:, 1].to_numpy()

                elapsed = time.perf_counter() - series_start
                series_info.update(
                    {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "aic": float(best_aic),
                        "fit_time_sec": elapsed,
                    }
                )
                LOGGER.info(
                    "[ARIMA][%s] order=%s seasonal=%s AIC=%.2f h=%d secs=%.2f",
                    item_id,
                    order,
                    seasonal_order,
                    best_aic,
                    prediction_length,
                    elapsed,
                )

        if fc_mean_vals is None:
            raise RuntimeError("Forecast mean missing; this is a bug in fit_predict_arima_ebs")

        if lower_vals is None or upper_vals is None:
            lower_vals = fc_mean_vals * 0.9
            upper_vals = fc_mean_vals * 1.1

        if enforce_nonneg:
            fc_mean_vals = np.clip(fc_mean_vals, 0.0, None)
            lower_vals = np.clip(lower_vals, 0.0, None)
            upper_vals = np.clip(upper_vals, 0.0, None)

        if round_to_int:
            fc_mean_vals = np.rint(fc_mean_vals)
            lower_vals = np.rint(lower_vals)
            upper_vals = np.rint(upper_vals)

        forecast_index = pd.date_range(target.index[-1] + freq_offset, periods=prediction_length, freq=freq_offset)
        forecast_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": forecast_index,
                "p10": lower_vals,
                "p50": fc_mean_vals,
                "p90": upper_vals,
            }
        )

        results.append(_SeriesResult(item_id=item_id, forecast=forecast_df, info=series_info))

    if not results:
        raise ValueError("No series found in dataframe")

    forecast_concat = pd.concat([res.forecast for res in results], ignore_index=True)
    forecast_concat.sort_values(["item_id", "timestamp"], inplace=True)
    forecast_concat.reset_index(drop=True, inplace=True)

    info: Dict[str, Any] = {
        "series": [res.info for res in results],
        "fallback_count": fallback_count,
        "conf_level": conf_level,
    }
    if metrics_mae:
        info["metrics"] = {"mae_holdout": metrics_mae}

    # Aggregate diagnostics for logging convenience
    mae_values = list(metrics_mae.values())
    overall_mae = float(np.mean(mae_values)) if mae_values else float("nan")
    peak_day: str | None = None
    peak_item: str | None = None
    if not forecast_concat.empty:
        peak_idx = int(forecast_concat["p50"].idxmax())
        peak_ts_raw: Any = forecast_concat.loc[peak_idx, "timestamp"]
        peak_ts = pd.Timestamp(cast(Any, peak_ts_raw))
        peak_day = peak_ts.strftime("%Y-%m-%d")
        peak_item = str(forecast_concat.loc[peak_idx, "item_id"])
        peak_value = float(np.asarray(forecast_concat.loc[peak_idx, "p50"]).item())
        info["peak"] = {"item_id": peak_item, "date": peak_day, "value": peak_value}

    month_end_flag = any(res.info.get("month_end_windows") for res in results)
    payroll_flag = any(res.info.get("payroll_dates") for res in results)
    info["events"] = {"month_end": month_end_flag, "payroll": payroll_flag}

    total_elapsed = time.perf_counter() - total_start
    LOGGER.info(
        "[ARIMA] series=%d | conf_level=%.2f | fallback=%d | elapsed=%.2fs",
        len(results),
        conf_level,
        fallback_count,
        total_elapsed,
    )

    mae_label = "n/d" if math.isnan(overall_mae) else f"{overall_mae:.2f}"
    peak_label = peak_day or "n/d"
    LOGGER.info(
        "[EBS] MAE_holdout=%s | peak_day=%s (%s) | bands=month-end/payroll=%s/%s",
        mae_label,
        peak_label,
        peak_item or "n/a",
        month_end_flag,
        payroll_flag,
    )

    return forecast_concat, info
