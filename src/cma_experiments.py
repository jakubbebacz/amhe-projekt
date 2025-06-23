import argparse
import os
import time
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cmaes import CMA
from scipy.stats import wilcoxon

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)


def rastrigin(x: np.ndarray) -> float:
    n = x.size
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))


def ackley(x: np.ndarray, a=20, b=0.2, c=2 * np.pi) -> float:
    n = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return float(term1 + term2 + a + np.e)


def schwefel(x: np.ndarray) -> float:
    n = x.size
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


FUNCTIONS: Dict[str, Callable[[np.ndarray], float]] = {
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "schwefel": schwefel,
}

OPTIMUM_VALUES = {
    "rosenbrock": 0,
    "rastrigin": 0,
    "ackley": 0,
    "schwefel": 0,
}


def run_cma_es(
    func,
    func_name,
    dim,
    init_sigma,
    init_ps_random,
    base_seed,
    generator_name,
    max_evals,
    tol,
    num_runs,
):
    if generator_name == "mersenne":
        cma_seed = base_seed
        ps_rng = np.random.RandomState(base_seed + num_runs)
        randn_for_ps = ps_rng.randn
    else:
        seeder = np.random.PCG64(base_seed)
        cma_seed = seeder.spawn(1)[0]
        ps_rng = np.random.Generator(seeder.spawn(1)[0])
        randn_for_ps = ps_rng.standard_normal

    es = CMA(mean=np.zeros(dim), sigma=init_sigma, seed=cma_seed)
    popsize = es._popsize

    if init_ps_random:
        es._p_sigma = randn_for_ps(dim).astype(np.float64)

    optimum = OPTIMUM_VALUES.get(func_name, 0)

    evals = 0
    best_fitness = float("inf")
    history = []
    while not es.should_stop() and evals < max_evals:
        solutions = [es.ask() for _ in range(popsize)]
        evaluated = [(x, func(x)) for x in solutions]
        fitnesses = [f for (_, f) in evaluated]
        es.tell(evaluated)

        evals += popsize
        current_best = min(fitnesses)
        if current_best < best_fitness:
            best_fitness = current_best

        history.append({"evals": evals, "best_fitness": best_fitness})

        if abs(best_fitness - optimum) <= tol:
            break

    return {
        "best_fitness": best_fitness,
        "evals": evals,
        "converged": abs(best_fitness - optimum) <= tol,
        "history": history,
    }


def run_experiments(
    output_dir: str, functions: Dict, dims: List[int], num_runs: int, **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    records = []
    history_data = []
    generator_names = ["mersenne", "pcg"]
    versions = {"standard": False, "random": True}

    for generator_name in generator_names:
        for version_name, init_ps_random in versions.items():
            for func_name, func in functions.items():
                for dim in dims:
                    print(
                        f"Running: {generator_name}/{version_name}/{func_name}/dim={dim}..."
                    )
                    for seed in range(num_runs):
                        start = time.time()
                        res = run_cma_es(
                            func=func,
                            func_name=func_name,
                            dim=dim,
                            init_ps_random=init_ps_random,
                            base_seed=seed,
                            generator_name=generator_name,
                            num_runs=num_runs,  # <<< POPRAWKA
                            **kwargs,
                        )
                        end = time.time()
                        records.append(
                            {
                                "generator": generator_name,
                                "version": version_name,
                                "function": func_name,
                                "dim": dim,
                                "seed": seed,
                                "best_fitness": res["best_fitness"],
                                "evals": res["evals"],
                                "converged": res["converged"],
                                "time": end - start,
                            }
                        )
                        for h in res["history"]:
                            history_data.append(
                                {
                                    "generator": generator_name,
                                    "version": version_name,
                                    "function": func_name,
                                    "dim": dim,
                                    **h,
                                }
                            )

    df = pd.DataFrame.from_records(records)
    history_df = pd.DataFrame.from_records(history_data)

    df.to_csv(f"{output_dir}/results.csv", index=False)
    history_df.to_csv(f"{output_dir}/history.csv", index=False)

    return df, history_df


def analyze_results(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    summary = []
    for func_name in df["function"].unique():
        for gen_name in df["generator"].unique():
            for dim in df["dim"].unique():
                df_sub = df[
                    (df["function"] == func_name)
                    & (df["generator"] == gen_name)
                    & (df["dim"] == dim)
                ]
                pivot = df_sub.pivot(index="seed", columns="version", values="evals")
                if "standard" not in pivot.columns or "random" not in pivot.columns:
                    continue
                if (pivot["standard"] == pivot["random"]).all():
                    stat, p = 0.0, 1.0
                else:
                    try:
                        stat, p = wilcoxon(pivot["standard"], pivot["random"])
                    except ValueError:
                        stat, p = 0.0, 1.0
                summary.append(
                    {
                        "function": func_name,
                        "generator": gen_name,
                        "dim": dim,
                        "stat": stat,
                        "p_value": p,
                        "significant": p < alpha if not np.isnan(p) else False,
                    }
                )
    return pd.DataFrame(summary)


def plot_convergence(history_df: pd.DataFrame, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    for func_name in history_df["function"].unique():
        func_df = history_df[history_df["function"] == func_name]
        dims = sorted(func_df["dim"].unique())
        ncols = len(dims)
        fig, axes = plt.subplots(
            1, ncols, figsize=(5 * ncols, 5), sharey=True, squeeze=False
        )
        axes = axes.flatten()
        fig.suptitle(f"Konwergencja dla funkcji {func_name.capitalize()}", fontsize=16)
        for i, dim in enumerate(dims):
            ax = axes[i]
            dim_df = func_df[func_df["dim"] == dim]
            sns.lineplot(
                data=dim_df,
                x="evals",
                y="best_fitness",
                hue="version",
                style="generator",
                estimator="median",
                errorbar=("pi", 50),
                ax=ax,
            )
            ax.set_title(f"Wymiar = {dim}")
            ax.set_xlabel("Liczba ewaluacji funkcji")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="-", alpha=0.2)
            if i == 0:
                ax.set_ylabel("Najlepsza wartość funkcji (mediana)")
            handles, labels = ax.get_legend_handles_labels()
            label_map = {
                "version": "Wersja",
                "standard": "Zerowa",
                "random": "Losowa",
                "generator": "Generator",
                "mersenne": "Mersenne",
                "pcg": "PCG64",
            }
            new_labels = [label_map.get(l, l.capitalize()) for l in labels]
            from collections import OrderedDict

            handles_labels = OrderedDict(zip(new_labels, handles))
            ax.legend(handles_labels.values(), handles_labels.keys())
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{output_dir}/convergence_{func_name}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMA-ES experiment containerized")
    parser.add_argument("runs", type=int, help="Number of runs per combination")
    args = parser.parse_args()

    exp_params = {
        "init_sigma": 0.3,
        "max_evals": 10000,
        "tol": 1e-7,
    }

    df, history_df = run_experiments(
        output_dir="data",
        functions=FUNCTIONS,
        dims=[2, 10, 30],
        num_runs=args.runs,
        **exp_params,
    )

    summary = analyze_results(df)
    summary.to_csv("data/summary.csv", index=False)

    plot_convergence(history_df, "plots")

    print("Eksperyment zakończony.")
