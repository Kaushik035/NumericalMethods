import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Utility Functions
# -------------------------

def ensure_monotonic_time(t: np.ndarray, y: np.ndarray):
    """Ensure time is sorted and unique; reorders y accordingly."""
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]
    # drop duplicates
    _, idx = np.unique(t_sorted, return_index=True)
    return t_sorted[idx].astype(float), y_sorted[idx].astype(float)


def estimate_dt(t: np.ndarray) -> float:
    """Estimate delta t (sampling interval)."""
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs))


def rmse(y_true, y_pred):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def save_plot(x, ys, labels, xlabel, ylabel, title, outpath):
    """Save plot with multiple series."""
    plt.figure(figsize=(8, 5))
    for y, lbl in zip(ys, labels):
        plt.plot(x, y, label=lbl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[saved] {outpath}")


def E_theo_cstr(time, tau):
    return np.where(time >= 0.0, (1.0/tau)*np.exp(-time/tau), 0.0)

def F_theo_cstr(time, tau):
    return np.where(time >= 0.0, 1.0 - np.exp(-time/tau), 0.0)

def F_theo_lfr(time, tau):
    return np.where(time < 0.5*tau, 0.0, 1.0 - (tau**2)/(4.0*np.maximum(time,1e-12)**2))

def E_theo_lfr(time, tau):
    return np.where(time < 0.5*tau, 0.0, (tau**2)/(2.0*np.maximum(time,1e-12)**3))

def series_theory_on_grid(time, tau_L, tau_C):
    """Return (E_series_theo, F_series_theo) for LFR→CSTR on the given time grid."""
    dt = estimate_dt(time)
    # build individual theoretical curves on the same grid
    E_L = E_theo_lfr(time, tau_L)
    E_C = E_theo_cstr(time, tau_C)
    F_L = F_theo_lfr(time, tau_L)

    # Convolutions on a causal, uniform grid; trim back to grid length
    E_series = np.convolve(E_L, E_C) * dt
    E_series = E_series[:len(time)]

    # Step response of the series: F_series = F_LFR * E_CSTR
    F_series = np.convolve(F_L, E_C) * dt
    F_series = np.clip(F_series[:len(time)], 0.0, 1.0)
    return E_series, F_series, (E_L, E_C, F_L)



# -------------------------
# Pulse Input (CSTR only)
# -------------------------

def pulse_analysis(time, Cout):
    """
    Compute Eexp, mean residence time, variance, and CSTR theory E(t).
    Converts conductivity from microSiemens to Siemens before processing.
    """
    time = time.astype(float)
    Cout = Cout.astype(float) * 1e-6  # µS -> S

    area = np.trapezoid(Cout, time)
    if area <= 0:
        raise ValueError("Invalid area under C-curve, check data!")

    E_exp = Cout / area
    t_mean = np.trapezoid(time * E_exp, time)
    variance = np.trapezoid((time - t_mean) ** 2 * E_exp, time)

    # Theoretical CSTR response with τ = t_mean
    E_theo = (1.0 / t_mean) * np.exp(-time / t_mean) if t_mean > 0 else np.zeros_like(time)

    return E_exp, E_theo, float(t_mean), float(variance)


# -------------------------
# Step Input (LFR / LFR+CSTR)
# -------------------------

def step_analysis(time, Cout):
    """
    Compute Fexp, Eexp, theoretical Ftheo for LFR (per manual), and mean residence time.
    - Converts conductivity from microSiemens to Siemens.
    - Uses piecewise F_theo without mask.
    """
    time = time.astype(float)
    Cout = Cout.astype(float) * 1e-6  # µS -> S

    # Take C_ss as median of last 10% of points (robust to noise)
    n_tail = max(5, len(Cout) // 10)
    C_ss = float(np.median(Cout[-n_tail:]))
    if C_ss == 0:
        raise ValueError("Final conductivity C_ss is zero, cannot normalize!")

    # Normalize experimental step response
    F_exp = Cout / C_ss
    F_exp = np.clip(F_exp, 0.0, 1.2)

    # Differentiate to get E(t)
    dt = estimate_dt(time)
    E_exp = np.gradient(F_exp, dt)

    # Normalize E(t) so that ∫E dt = 1, then t_mean = ∫ t E dt
    E_nonneg = np.clip(E_exp, 0, None)
    areaE = np.trapezoid(E_nonneg, time)
    if areaE > 0:
        E_norm = E_nonneg / areaE
        t_mean = float(np.trapezoid(time * E_norm, time))
    else:
        t_mean = float("nan")

    # Theoretical F(t) for LFR (piecewise, per manual) with tau = t_mean
    if np.isfinite(t_mean) and t_mean > 0:
        tau = t_mean
        # piecewise: 0 for t < tau/2; else 1 - tau^2/(4 t^2)
        F_theo_LFR = np.where(
            time < 0.5 * tau,
            0.0,
            1.0 - (tau ** 2) / (4.0 * np.maximum(time, 1e-12) ** 2),
        )
        F_theo_LFR = np.clip(F_theo_LFR, 0.0, 1.0)
    else:
        F_theo_LFR = np.zeros_like(time)

    return F_exp, E_exp, F_theo_LFR, t_mean


# -------------------------
# Main Pipeline
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="RTD analysis from conductivity CSV files.")
    ap.add_argument("--pulse_csv", help="CSV file for Pulse input (CSTR)")
    ap.add_argument("--step_lfr_csv", help="CSV file for Step input (LFR)")
    ap.add_argument("--step_cstr_csv", help="CSV file for Step input (LFR+CSTR)")
    ap.add_argument("--outdir", default="rtd_outputs", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {}

    # ---------------- Pulse Input ----------------
    if args.pulse_csv:
        df = pd.read_csv(args.pulse_csv)
        t, Cout = ensure_monotonic_time(df.iloc[:, 0].values, df.iloc[:, 1].values)

        E_exp, E_theo, t_mean, variance = pulse_analysis(t, Cout)

        # Plots
        save_plot(t, [Cout], ["CSTR Cout(t) [µS]"], "Time (s)", "Conductivity (µS/cm)",
                  "Pulse Input: CSTR C(t)", outdir / "pulse_C_curve.png")
        save_plot(t, [E_exp, E_theo], ["E_exp(t)", "E_theo(t) CSTR"],
                  "Time (s)", "E(t)", "Pulse Input: CSTR E(t)", outdir / "pulse_E_curve.png")

        summary["pulse_CSTR"] = {"t_mean_s": t_mean, "variance_s2": variance}

        pd.DataFrame({"time_s": t, "Cout_uS": Cout, "E_exp": E_exp, "E_theo": E_theo}).to_csv(
            outdir / "pulse_processed.csv", index=False
        )

    # ---------------- Step Input ----------------
    if args.step_lfr_csv and args.step_cstr_csv:
        # ---- LFR ----
        df_lfr = pd.read_csv(args.step_lfr_csv)
        t_lfr, Cout_lfr = ensure_monotonic_time(df_lfr.iloc[:, 0].values, df_lfr.iloc[:, 1].values)
        F_exp_lfr, E_exp_lfr, F_theo_lfr, tmean_lfr = step_analysis(t_lfr, Cout_lfr)

        # ---- LFR + CSTR ----
        df_cstr = pd.read_csv(args.step_cstr_csv)
        t_cstr, Cout_cstr = ensure_monotonic_time(df_cstr.iloc[:, 0].values, df_cstr.iloc[:, 1].values)
        F_exp_cstr, E_exp_cstr, _, tmean_cstr = step_analysis(t_cstr, Cout_cstr)

                # ===== Special Question: LFR + CSTR theory on the experimental grid =====
        # Use τ_L from step LFR mean, τ_C from pulse CSTR mean if available, else skip theory
        tau_L = float(tmean_lfr)
        tau_C_pulse = summary.get("pulse_CSTR", {}).get("t_mean_s", None) or summary.get("pulse_CSTR", {}).get("t_mean", None)

        if tau_C_pulse is not None and np.isfinite(tau_L) and tau_L > 0:
            tau_C = float(tau_C_pulse)

            # Build theoretical series E(t) and F(t) on the LFR+CSTR time grid
            E_series_theo, F_series_theo, (E_L_theo, E_C_theo, F_L_theo) = series_theory_on_grid(t_cstr, tau_L, tau_C)

            # Compare theory vs experiment for the series (step input)
            save_plot(
                t_cstr, [F_exp_cstr, F_series_theo],
                ["F_exp (LFR+CSTR)", "F_theo (LFR * CSTR)"],
                "Time (s)", "F(t)",
                "Step: LFR+CSTR — F_exp vs F_theo",
                outdir / "series_Fexp_vs_Ftheo.png"
            )

            # “What are E and F?” visualizers (theoretical components)
            save_plot(
                t_cstr, [E_L_theo, E_C_theo, E_series_theo],
                ["E_LFR (theo)", "E_CSTR (theo)", "E_series (conv)"],
                "Time (s)", "E(t)",
                "Theoretical E(t): LFR, CSTR, and series (convolution)",
                outdir / "series_E_components.png"
            )
            save_plot(
                t_cstr, [F_L_theo, F_theo_cstr(t_cstr, tau_C), F_series_theo],
                ["F_LFR (theo)", "F_CSTR (theo)", "F_series (LFR→CSTR)"],
                "Time (s)", "F(t)",
                "Theoretical F(t): LFR, CSTR, and series",
                outdir / "series_F_components.png"
            )

            # Optional: show E_exp for series alongside E_series_theo (from step via dF/dt it's noisier, but ok)
            save_plot(
                t_cstr, [E_exp_cstr, E_series_theo],
                ["E_exp (LFR+CSTR)", "E_theo (LFR * CSTR)"],
                "Time (s)", "E(t)",
                "LFR+CSTR — E_exp vs E_theo",
                outdir / "series_Eexp_vs_Etheo.png"
            )

            # Metric
            summary["series_LFR_CSTR_theory"] = {
                "tau_L_from_step_LFR_s": tau_L,
                "tau_C_from_pulse_CSTR_s": tau_C,
                "RMSE_Fexp_vs_Ftheo": float(rmse(F_exp_cstr, F_series_theo))
            }
        else:
            # If you didn’t run --pulse_csv earlier, theory for the series is skipped
            summary["series_LFR_CSTR_theory"] = {
                "note": "No pulse_CSTR t_mean available; run with --pulse_csv to enable series theory."
            }


        # Plots
        save_plot(t_lfr, [Cout_lfr], ["LFR Cout(t) [µS]"], "Time (s)", "Conductivity (µS/cm)",
                  "Step Input: LFR Cout(t)", outdir / "step_LFR_C.png")
        save_plot(t_cstr, [Cout_cstr], ["LFR+CSTR Cout(t) [µS]"], "Time (s)", "Conductivity (µS/cm )",
                  "Step Input: LFR+CSTR Cout(t)", outdir / "step_LFR_CSTR_C.png")

        save_plot(t_lfr, [E_exp_lfr], ["E_exp LFR"], "Time (s)", "E(t)",
                  "Step Input: E(t) LFR", outdir / "step_LFR_E.png")
        save_plot(t_cstr, [E_exp_cstr], ["E_exp LFR+CSTR"], "Time (s)", "E(t)",
                  "Step Input: E(t) LFR+CSTR", outdir / "step_LFR_CSTR_E.png")

        save_plot(t_lfr, [F_exp_lfr, F_theo_lfr], ["F_exp LFR", "F_theo LFR"],
                  "Time (s)", "F(t)", "Step Input: F(t) LFR", outdir / "step_LFR_F.png")
        save_plot(t_cstr, [F_exp_cstr], ["F_exp LFR+CSTR"],
                  "Time (s)", "F(t)", "Step Input: F(t) LFR+CSTR", outdir / "step_LFR_CSTR_F.png")

        # Metrics (RMSE across whole series is fine per module)
        rmse_val = rmse(F_exp_lfr, F_theo_lfr)

        summary["step_LFR"] = {"t_mean_s": tmean_lfr, "RMSE_vs_theory": rmse_val}
        summary["step_LFR+CSTR"] = {"t_mean_s": tmean_cstr}

        # Save processed CSVs
        pd.DataFrame({"time_s": t_lfr, "Cout_uS": Cout_lfr, "F_exp": F_exp_lfr,
                      "E_exp": E_exp_lfr, "F_theo": F_theo_lfr}).to_csv(
            outdir / "step_LFR_processed.csv", index=False
        )
        pd.DataFrame({"time_s": t_cstr, "Cout_uS": Cout_cstr, "F_exp": F_exp_cstr,
                      "E_exp": E_exp_cstr}).to_csv(
            outdir / "step_LFR_CSTR_processed.csv", index=False
        )

    # ---------------- Save Summary ----------------
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[done] Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
