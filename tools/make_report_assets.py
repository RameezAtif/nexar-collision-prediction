"""Generate the report's figures and LaTeX tables from the result CSVs (no training, no raw data).

Every number in the report's tables and figures comes from the files read here, so the report is traceable to the repository:

    results/temporal_ablation_summary.csv      mean / std over 5 seeds per model, variant and TTE   (notebook 06)
    results/temporal_ablation_bootstrap.csv    paired bootstrap CIs over the 300 test clips          (notebook 06)
    results/failure_frame_position_ap.csv      single-frame AP at each frame position                (notebook 08)
    results/failure_alert_timing.csv           AUC of positives before / after the annotated alert   (notebook 08)

Usage (from the project root):  python tools/make_report_assets.py
Writes report/figures/*.pdf and report/tables/*.tex.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
OUT_FIG = Path("report/figures")
OUT_TAB = Path("report/tables")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

TTES = [0.5, 1.0, 1.5]
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
})

summary = pd.read_csv(RESULTS / "temporal_ablation_summary.csv")
boot = pd.read_csv(RESULTS / "temporal_ablation_boot" "strap.csv")
position = pd.read_csv(RESULTS / "failure_frame_position_ap.csv")
alert = pd.read_csv(RESULTS / "failure_alert_timing.csv")


def stat(variant, tte, col):
    r = summary[(summary["variant"] == variant) & (summary["tte"] == tte)].iloc[0]
    return r[f"{col}_mean"], r[f"{col}_std"]


def pm(variant, tte, col="AP", digits=3):
    m, s = stat(variant, tte, col)
    return f"{m:.{digits}f}" if np.isnan(s) else f"{m:.{digits}f}\\,$\\pm$\\,{s:.{digits}f}"


# ---------------------------------------------------------------- Figure 1: AP vs TTE + single-frame position
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35), gridspec_kw={"width_ratios": [1, 1.15]})

ax = axes[0]
ax.axhline(0.5, color=GREY, ls=(0, (2, 2)), lw=1)
for variant, colour, label, marker in [("lstm", BLUE, "CNN+LSTM", "o"), ("last_frame", INK2, "last frame only", "s"),
                                       ("optical_flow", ORANGE, "optical flow", "^")]:
    m = np.array([stat(variant, t, "AP")[0] for t in TTES])
    s = np.nan_to_num(np.array([stat(variant, t, "AP")[1] for t in TTES]))
    ax.errorbar(TTES, m, yerr=s, color=colour, marker=marker, ms=4.5, lw=1.5, capsize=2, elinewidth=0.9, label=label)
ax.text(1.5, 0.505, "naive / chance", ha="right", va="bottom", color=INK2, fontsize=6.5)
ax.set_xticks(TTES)
ax.set_xlim(0.4, 1.6)
ax.set_ylim(0.45, 0.82)
ax.set_xlabel("time to event, TTE (s)")
ax.set_ylabel("average precision (AP)")
ax.set_title("(a) AP versus prediction horizon", loc="left")
ax.legend(loc="upper right")

ax = axes[1]
ax.axhline(0.5, color=GREY, ls=(0, (2, 2)), lw=1)
for tte, colour, marker in zip(TTES, [BLUE, ORANGE, "#1baf7a"], ["o", "^", "s"]):
    sub = position[position["tte"] == tte].sort_values("seconds_before_event")
    ax.errorbar(sub["seconds_before_event"], sub["AP_mean"], yerr=sub["AP_std"], color=colour, marker=marker, ms=3, lw=1.1,
                capsize=1.5, elinewidth=0.7, label=f"clip ending {tte} s before event")
ax.invert_xaxis()
ax.set_xlabel("seconds before the event at which the single frame was taken")
ax.set_ylabel("single-frame AP")
ax.set_ylim(0.45, 0.82)
ax.set_title("(b) one frame, by its distance from the event", loc="left")
ax.legend(loc="upper left")
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(OUT_FIG / "fig_main.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: paired differences (single-panel forest plot)
rows = [("optical_flow", "optical flow"), ("last_frame", "last frame only"), ("mean_pool", "mean-pool (no order)"),
        ("lstm_train_shuffled", "LSTM, shuffled train+test"), ("lstm_test_shuffled", "LSTM, shuffled at test only")]
NAME = {"optical_flow": "Optical flow", "last_frame": "Last frame (no temporal)", "mean_pool": "Mean-pool (no order)",
        "lstm_train_shuffled": "LSTM, shuffled train+test", "lstm_test_shuffled": "LSTM, shuffled at test only"}
fig, ax = plt.subplots(figsize=(3.45, 2.5))
ax.axvline(0, color=INK2, ls=(0, (2, 2)), lw=0.9)
offsets = {0.5: 0.24, 1.0: 0.0, 1.5: -0.24}
markers = {0.5: "o", 1.0: "^", 1.5: "s"}
for k, (variant, _) in enumerate(rows):
    y0 = len(rows) - 1 - k
    for tte in TTES:
        r = boot[(boot["comparison"] == f"AP diff: CNN+LSTM − {NAME[variant]}") & (boot["tte"] == tte)].iloc[0]
        sig = bool(r["excludes_zero"])
        ax.errorbar(r["estimate"], y0 + offsets[tte], xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                    fmt=markers[tte], ms=3.6, color=BLUE if sig else GREY, mfc=BLUE if sig else "white", mew=0.9, capsize=1.2, elinewidth=0.9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([lbl for _, lbl in rows][::-1])
ax.grid(axis="y", visible=False)
ax.set_xlabel("AP difference: CNN+LSTM minus comparison")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker=markers[t], color=INK2, ls="", ms=3.6, label=f"TTE {t} s") for t in TTES],
          loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT_FIG / "fig_paired.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure: clip construction (timeline)
fig, ax = plt.subplots(figsize=(3.45, 1.75))
ALERT = -1.6   # mean annotated lead time of the alert before the event (seconds)
for k, tte in enumerate(TTES):
    y = len(TTES) - 1 - k
    end = -tte
    ax.add_patch(plt.Rectangle((end - 2.0, y - 0.28), 2.0, 0.56, color=BLUE, alpha=0.18, lw=0))
    ax.vlines(np.linspace(end - 2.0, end, 16), y - 0.28, y + 0.28, color=BLUE, lw=0.5)
    ax.plot([end], [y], marker="v", color=INK, ms=4)
    ax.annotate(f"TTE {tte} s", (end - 2.0, y), xytext=(-3, 0), textcoords="offset points", va="center", ha="right", fontsize=6.5, color=INK)
ax.axvline(0, color=ORANGE, lw=1.6)
ax.axvline(ALERT, color=GREY, lw=1, ls=(0, (3, 2)))
ax.text(0.04, len(TTES) - 0.35, "event", color=ORANGE, fontsize=7, va="bottom")
ax.text(ALERT - 0.04, len(TTES) - 0.35, "mean alert", color=INK2, fontsize=6.3, va="bottom", ha="right")
ax.set_xlim(-5.0, 0.9)
ax.set_ylim(-0.6, len(TTES) - 0.1 + 0.3)
ax.set_yticks([])
ax.set_xlabel("time relative to the annotated event (s)")
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(OUT_FIG / "fig_clips.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure: alert timing (CNN+LSTM)
fig, ax = plt.subplots(figsize=(3.45, 1.9))
ax.axvline(0.5, color=GREY, ls=(0, (2, 2)), lw=1)
ylabels = []
for k, tte in enumerate(TTES):
    for g, (grp, hollow) in enumerate([("ends after alert", False), ("ends before alert", True)]):
        r = alert[(alert["model"] == "lstm") & (alert["tte"] == tte) & (alert["group"] == grp)].iloc[0]
        y = (len(TTES) - 1 - k) * 2.6 + (1 if g == 0 else 0)
        ax.errorbar(r["AUC_vs_negatives"], y, xerr=[[r["AUC_vs_negatives"] - r["ci_low"]], [r["ci_high"] - r["AUC_vs_negatives"]]], fmt="o", ms=4,
                    color=BLUE, mfc="white" if hollow else BLUE, mew=1, capsize=1.5, elinewidth=1)
        ax.text(0.99, y, f"n={int(r['n_positives'])}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6, color=INK2)
        ylabels.append((y, f"{tte} s, clip ends {'after' if g == 0 else 'before'} alert"))
ax.set_yticks([y for y, _ in ylabels])
ax.set_yticklabels([l for _, l in ylabels], fontsize=6)
ax.set_xlim(0.3, 0.9)
ax.set_xlabel("AUC of these collision clips against all normal clips")
ax.grid(axis="y", visible=False)
fig.tight_layout(pad=0.3)
fig.savefig(OUT_FIG / "fig_alert.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Table I: main results
lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{AP} & \multicolumn{3}{c}{Recall at FPR $\leq 10\%$} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Model & 0.5\,s & 1.0\,s & 1.5\,s & 0.5\,s & 1.0\,s & 1.5\,s \\", r"\midrule"]
for variant, name in [("naive", "Naive (Model 0)"), ("optical_flow", "Optical flow (Model 1)"), ("lstm", "CNN+LSTM (Model 2)")]:
    ap = " & ".join(pm(variant, t, "AP") for t in TTES)
    rc = " & ".join(pm(variant, t, "Recall@FPR10", 2) for t in TTES)
    lines.append(f"{name} & {ap} & {rc} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT_TAB / "table_main.tex").write_text("\n".join(lines) + "\n", encoding="utf8")

# ---------------------------------------------------------------- Table II: temporal ablation
lines = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Variant (same features) & 0.5\,s & 1.0\,s & 1.5\,s \\", r"\midrule"]
for variant, name in [("last_frame", "Last frame only"), ("mean_pool", "Mean-pool, no order"),
                      ("lstm_train_shuffled", "LSTM, shuffled train+test"), ("lstm_test_shuffled", "LSTM, shuffled at test only"),
                      ("lstm", "CNN+LSTM (frame order kept)")]:
    cells = []
    for t in TTES:
        cell = pm(variant, t, "AP")
        if variant != "lstm":
            r = boot[(boot["comparison"] == f"AP diff: CNN+LSTM − {NAME[variant]}") & (boot["tte"] == t)].iloc[0]
            if bool(r["excludes_zero"]):
                cell += r"$^{\dagger}$"
        cells.append(cell)
    lines.append(f"{name} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT_TAB / "table_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf8")

# ---------------------------------------------------------------- Table: epoch-selection pitfall
nb05 = pd.read_csv(RESULTS / "cnn_lstm_results.csv").sort_values("tte")
chk = pd.read_csv(RESULTS / "selection_check_fulldata.csv")
g = chk.groupby("tte").agg(final=("final_epoch", "mean"), final_sd=("final_epoch", "std"), best=("best_epoch", "mean"))
lines = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"CNN+LSTM, how the epoch is chosen & 0.5\,s & 1.0\,s & 1.5\,s \\", r"\midrule"]
lines.append("Best epoch on the test videos (first run) & " + " & ".join(f"{v:.3f}" for v in nb05["AP"]) + r" \\")
lines.append("Same, mean of 5 seeds & " + " & ".join(f"{g.loc[t, 'best']:.3f}" for t in TTES) + r" \\")
lines.append("Fixed 10 epochs, no selection & " + " & ".join(f"{g.loc[t, 'final']:.3f}" + r"\,$\pm$\," + f"{g.loc[t, 'final_sd']:.3f}" for t in TTES) + r" \\")
lines.append("Epoch chosen on held-out training videos & " + " & ".join(pm("lstm", t, "AP") for t in TTES) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT_TAB / "table_selection.tex").write_text("\n".join(lines) + "\n", encoding="utf8")

# ---------------------------------------------------------------- Numbers quoted in the text (printed and saved for checking)
def drop(variant, far):
    r = boot[(boot["comparison"] == f"AP drop 0.5s→{far}s: {NAME.get(variant, 'CNN + LSTM')}") & (boot["tte"] == far)].iloc[0]
    return r["estimate"], r["ci_low"], r["ci_high"]


facts = {}
for t in TTES:
    r = boot[(boot["comparison"] == "AP diff: CNN+LSTM − Optical flow") & (boot["tte"] == t)].iloc[0]
    facts[f"lstm_minus_flow_{t}"] = (r["estimate"], r["ci_low"], r["ci_high"])
facts["lstm_drop_0.5_to_1.5"] = drop("lstm", 1.5)
facts["lstm_drop_0.5_to_1.0"] = drop("lstm", 1.0)
facts["flow_drop_0.5_to_1.5"] = drop("optical_flow", 1.5)
a = alert.copy()
for model in ["lstm", "last_frame"]:
    for t in TTES:
        for g in ["ends before alert", "ends after alert"]:
            s = a[(a["model"] == model) & (a["tte"] == t) & (a["group"] == g)]
            if len(s):
                facts[f"alert_auc_{model}_{t}_{g.split()[1]}"] = (s.iloc[0]["AUC_vs_negatives"], s.iloc[0]["ci_low"], s.iloc[0]["ci_high"], int(s.iloc[0]["n_positives"]))
pos_far = position[position["seconds_before_event"] >= 3.0]
facts["single_frame_AP_3s_or_more_before_event"] = (pos_far["AP_mean"].mean(), pos_far["AP_mean"].min(), pos_far["AP_mean"].max())
pd.Series({k: tuple(round(float(x), 3) for x in v) for k, v in facts.items()}).to_csv(OUT_TAB / "numbers_quoted_in_text.csv", header=["values"])
print("wrote", sorted(p.name for p in OUT_FIG.glob("*.pdf")), sorted(p.name for p in OUT_TAB.glob("*")))
for k, v in facts.items():
    print(f"  {k}: {tuple(round(float(x), 3) for x in v)}")
