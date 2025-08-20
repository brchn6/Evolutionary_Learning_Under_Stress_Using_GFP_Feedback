"""
Batch runner for recommended protocols.
Generates plots and brief interpretations into ./plots, plus a summary report.
"""
import os
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless backend for saving figures
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so `src` is importable when running this script
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.main import (
    SimulationParams, GFPParams,
    run_evolution_experiment, calculate_learning_metrics,
)

PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_plots(results, metrics, tag: str):
    driver = results['driver']
    times = np.array(driver['time'])
    temps = np.array(driver['temperature'])
    gfp_driver = np.array(driver['mean_gfp'])
    gfp_c30 = np.array(results['control_30']['mean_gfp'])
    gfp_c39 = np.array(results['control_39']['mean_gfp'])

    # Temperature plot
    plt.figure(figsize=(7,4))
    plt.plot(times, temps, label='Driver Temp', lw=2)
    plt.axhline(30, ls='--', c='tab:red', label='Control 30°C')
    plt.axhline(39, ls='--', c='tab:green', label='Control 39°C')
    plt.xlabel('Time (min)'); plt.ylabel('Temperature (°C)'); plt.title(f'Temperature - {tag}')
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"temp_{tag}.png", dpi=150)
    plt.close()

    # GFP plot
    plt.figure(figsize=(7,4))
    plt.plot(times, gfp_driver, label='Driver GFP', lw=2)
    plt.plot(times, gfp_c30, label='Control 30°C', lw=1)
    plt.plot(times, gfp_c39, label='Control 39°C', lw=1)
    if results['passives']:
        passive_mean = np.mean([np.array(p['mean_gfp']) for p in results['passives']], axis=0)
        plt.plot(times, passive_mean, label=f'Passives (n={len(results["passives"])})', lw=1, ls=':')
    plt.xlabel('Time (min)'); plt.ylabel('Mean GFP'); plt.title(f'GFP - {tag}')
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"gfp_{tag}.png", dpi=150)
    plt.close()

    # Save brief interpretation
    interp = []
    score = metrics.get('learning_score', 0)
    final_temp = metrics.get('final_temperature', 39)
    final_gfp = metrics.get('final_gfp', 0)
    interp.append(f"Learning score: {score:.2f}; Final temp: {final_temp:.1f}°C; Final driver GFP: {final_gfp:.1f}")
    if final_temp < 35:
        interp.append("Driver cooled substantially; indicates good feedback-driven adaptation.")
    elif final_temp < 37:
        interp.append("Moderate cooling; parameters allow some adaptation but not strong.")
    else:
        interp.append("Weak/no cooling; likely low sensitivity or high cost/noise.")
    if gfp_c39[-1] > gfp_c30[-1]:
        interp.append("39°C control > 30°C control, consistent with heat-induced switching/selection.")
    (PLOTS_DIR / f"interpretation_{tag}.txt").write_text("\n".join(interp), encoding='utf-8')


def run_one(tag: str, sim_kwargs: dict, gfp_kwargs: dict, mode: str):
    sim = SimulationParams(**sim_kwargs)
    gfp = GFPParams(**gfp_kwargs)
    results = run_evolution_experiment(sim, gfp, mode)
    metrics = calculate_learning_metrics(results)
    # Save metrics json
    (PLOTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    save_plots(results, metrics, tag)


def main():
    protocols = [
        # Basic Learning (Beginner)
        {
            'tag': 'basic_learning',
            'mode': 'continuous',
            'sim': dict(total_time=1000, population_size=200, num_passive_wells=3,
                        feedback_mode='linear', feedback_sensitivity=1.0,
                        temp_inertia=0.25, start_at_max_temp=True, metric_burn_in=10, random_seed=42),
            'gfp': dict(inherit_sd=5.0, switch_prob_base=0.01, cost_strength=0.3, cost_exponent=1.5),
        },
        # Binary Switching (Intermediate)
        {
            'tag': 'binary_switching',
            'mode': 'binary',
            'sim': dict(total_time=800, population_size=300, num_passive_wells=4,
                        feedback_mode='step', feedback_sensitivity=1.5,
                        temp_inertia=0.25, start_at_max_temp=True, metric_burn_in=10, random_seed=43),
            'gfp': dict(switch_prob_base=0.02, cost_strength=0.4, cost_exponent=1.0),
        },
        # Challenging Conditions (Advanced)
        {
            'tag': 'challenging',
            'mode': 'continuous',
            'sim': dict(total_time=1500, population_size=500, num_passive_wells=5,
                        feedback_mode='sigmoid', feedback_sensitivity=0.6,
                        temp_inertia=0.25, start_at_max_temp=True, metric_burn_in=10, random_seed=44),
            'gfp': dict(inherit_sd=8.0, switch_prob_base=0.008, cost_strength=0.5, cost_exponent=2.0),
        },
        # Failure Mode Analysis
        {
            'tag': 'failure_mode',
            'mode': 'continuous',
            'sim': dict(total_time=1000, population_size=150, num_passive_wells=3,
                        feedback_mode='linear', feedback_sensitivity=0.3,
                        temp_inertia=0.25, start_at_max_temp=True, metric_burn_in=10, random_seed=45),
            'gfp': dict(inherit_sd=15.0, switch_prob_base=0.003, cost_strength=0.8, cost_exponent=2.5),
        },
        # Rapid Learning (Expert)
        {
            'tag': 'rapid_learning',
            'mode': 'binary',
            'sim': dict(total_time=500, population_size=400, num_passive_wells=3,
                        feedback_mode='exponential', feedback_sensitivity=2.0,
                        temp_inertia=0.25, start_at_max_temp=True, metric_burn_in=10, random_seed=46),
            'gfp': dict(switch_prob_base=0.04, cost_strength=0.2, cost_exponent=1.0),
        },
    ]

    for p in protocols:
        print(f"Running protocol: {p['tag']}...")
        run_one(p['tag'], p['sim'], p['gfp'], p['mode'])
        print(f"Saved plots and metrics for: {p['tag']}")

    build_summary_report(protocols)
    print(f"\nSummary report written to: {PLOTS_DIR / 'summary.md'}")


def build_summary_report(protocols):
    lines = [
        "# Batch Summary: Recommended Protocols",
        "",
        "This report aggregates metrics and links to plots generated by scripts/run_protocols.py.",
        "",
    ]
    for p in protocols:
        tag = p['tag']
        metrics_path = PLOTS_DIR / f"metrics_{tag}.json"
        interp_path = PLOTS_DIR / f"interpretation_{tag}.txt"
        temp_img = f"temp_{tag}.png"
        gfp_img = f"gfp_{tag}.png"

        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            except Exception:
                metrics = {}
        interp_txt = interp_path.read_text(encoding='utf-8') if interp_path.exists() else ""

        lines += [
            f"## {tag}",
            "",
            "**Key metrics**:",
            "",
        ]
        if metrics:
            lines += [
                f"- learning_score: {metrics.get('learning_score', 0):.2f}",
                f"- final_gfp: {metrics.get('final_gfp', 0):.1f}",
                f"- final_temperature: {metrics.get('final_temperature', 0):.1f} °C",
                f"- adaptation_time: {metrics.get('adaptation_time', 'NA')}",
                f"- final_high_gfp_fraction: {metrics.get('final_high_gfp_fraction', 0):.2f}",
                "",
            ]
        else:
            lines += ["- (metrics missing)", ""]

        if interp_txt:
            lines += ["**Interpretation**:", "", f"> {interp_txt.replace('\n', '\n> ')}", ""]

        # Embed images (relative paths)
        lines += [
            f"![Temperature {tag}](./{temp_img})",
            "",
            f"![GFP {tag}](./{gfp_img})",
            "",
            "---",
            "",
        ]

    (PLOTS_DIR / "summary.md").write_text("\n".join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
