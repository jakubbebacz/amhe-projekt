import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# zamieniamy cma na cmaes
from cmaes import CMA
from scipy.stats import wilcoxon

# Ustawienie stylu wykresów
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


# benchmark functions (bez zmian)
def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)


def rastrigin(x):
    n = x.size
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    n = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return term1 + term2 + a + np.e


def schwefel(x):
    n = x.size
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


FUNCTIONS = {
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "schwefel": schwefel,
}


def run_cma_es(
    func, dim, init_sigma, init_ps_random, base_seed, generator_name, max_evals, tol
):
    # Inicjalizacja algorytmu
    es = CMA(mean=np.zeros(dim), sigma=init_sigma, seed=base_seed)
    popsize = es._popsize

    # Losowa inicjalizacja p_sigma
    if generator_name == "mersenne":
        rng = np.random.RandomState(base_seed)  # klasyczny Mersenne Twister
        randn = rng.randn  # zgodny z API: zwraca np.ndarray
    else:
        rng = np.random.Generator(np.random.PCG64(base_seed))  # nowoczesny PCG64
        randn = rng.standard_normal

    # Losowa inicjalizacja ścieżki ewolucyjnej (p_sigma)
    if init_ps_random:
        es._p_sigma = randn(dim).astype(np.float64)

    # Wartość optimum
    optimum = {
        "rosenbrock": 0,
        "rastrigin": 0,
        "ackley": 0,
        "schwefel": 418.9829 * dim,
    }.get(func.__name__, 0)

    evals = 0
    best_fitness = float("inf")
    history = []

    while not es.should_stop() and evals < max_evals:
        # Generowanie populacji
        solutions = [es.ask() for _ in range(popsize)]

        # Ewaluacja populacji
        evaluated = [(x, func(x)) for x in solutions]
        fitnesses = [f for (_, f) in evaluated]

        # Aktualizacja algorytmu
        es.tell(evaluated)

        # Statystyki
        evals += popsize
        best_fitness = min(best_fitness, min(fitnesses))
        history.append({"evals": evals, "best_fitness": best_fitness})

        if abs(best_fitness - optimum) <= tol:
            break

    return {
        "best_fitness": best_fitness,
        "evals": evals,
        "converged": abs(best_fitness - optimum) <= tol,
        "history": history,
    }


def run_experiments(output_csv, functions, dims, num_runs, init_sigma, max_evals, tol):
    records = []
    history_data = []
    generator_names = ["mersenne", "pcg"]
    versions = {"standard": False, "random": True}

    for generator_name in generator_names:
        for version_name, init_ps_random in versions.items():
            for func_name, func in functions.items():
                for dim in dims:
                    for seed in range(num_runs):
                        start = time.time()
                        res = run_cma_es(
                            func,
                            dim,
                            init_sigma,
                            init_ps_random,
                            seed,
                            generator_name,
                            max_evals,
                            tol,
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
    df.to_csv(output_csv, index=False)
    history_df = pd.DataFrame.from_records(history_data)
    history_csv = os.path.splitext(output_csv)[0] + "_history.csv"
    history_df.to_csv(history_csv, index=False)
    return df, history_df


# stats
def analyze_results(df, alpha=0.05):
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

                # Handle cases where all values are identical
                if (pivot["standard"] == pivot["random"]).all():
                    stat = 0.0
                    p = np.nan
                else:
                    stat, p = wilcoxon(pivot["standard"], pivot["random"])

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


# Plotting functions
def plot_convergence(history_df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    for func_name in history_df["function"].unique():
        plt.figure(figsize=(10, 6))
        func_df = history_df[history_df["function"] == func_name]

        agg_df = (
            func_df.groupby(["dim", "version", "evals"])["best_fitness"]
            .agg(
                [
                    "median",
                    lambda x: np.quantile(x, 0.25),
                    lambda x: np.quantile(x, 0.75),
                ]
            )
            .reset_index()
        )
        agg_df.columns = ["dim", "version", "evals", "median", "q25", "q75"]

        # subplots for each dimension
        dims = sorted(agg_df["dim"].unique())
        fig, axes = plt.subplots(1, len(dims), figsize=(15, 5), sharey=True)
        fig.suptitle(f"Konwergencja dla funkcji {func_name.capitalize()}", fontsize=16)

        if len(dims) == 1:
            axes = [axes]

        for i, dim in enumerate(dims):
            ax = axes[i]
            dim_df = agg_df[agg_df["dim"] == dim]

            for version, color in zip(["standard", "random"], ["#1f77b4", "#ff7f0e"]):
                version_df = dim_df[dim_df["version"] == version]
                ax.plot(
                    version_df["evals"],
                    version_df["median"],
                    label=f"Inicjalizacja {'losowa' if version == 'random' else 'zerowa'}",
                    color=color,
                )
                ax.fill_between(
                    version_df["evals"],
                    version_df["q25"],
                    version_df["q75"],
                    alpha=0.2,
                    color=color,
                )

            ax.set_title(f"Wymiar = {dim}")
            ax.set_xlabel("Liczba ewaluacji funkcji")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="-", alpha=0.2)
            if i == 0:
                ax.set_ylabel("Najlepsza wartość funkcji")
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{output_dir}/convergence_{func_name}.png")
        plt.close()


def plot_summary(summary_df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    plot_df = summary_df.copy()
    plot_df["dim"] = plot_df["dim"].astype(str)
    plot_df["significant"] = plot_df["significant"].map({True: "Tak", False: "Nie"})

    # Create plot
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        data=plot_df,
        x="dim",
        y="function",
        hue="significant",
        style="generator",
        s=200,
        palette={"Tak": "red", "Nie": "green"},
    )

    # Add p-value annotations
    for i, row in plot_df.iterrows():
        p_text = f"p={row['p_value']:.3f}" if not np.isnan(row["p_value"]) else "p=NaN"
        ax.text(
            row["dim"],
            row["function"],
            p_text,
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            bbox=dict(
                facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"
            ),
        )

    plt.title("Istotność statystyczna różnic między wersjami algorytmu", fontsize=16)
    plt.xlabel("Wymiar problemu")
    plt.ylabel("Funkcja testowa")
    plt.legend(title="Istotna różnica", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/statistical_significance.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMA-ES experiment containerized")
    parser.add_argument("runs", type=int, help="Number of runs per combination")
    args = parser.parse_args()

    dims = [2, 10, 30]
    init_sigma = 0.3
    max_evals = 10000
    tol = 1e-7

    df, history_df = run_experiments(
        output_csv="data/results.csv",
        functions=FUNCTIONS,
        dims=dims,
        num_runs=args.runs,
        init_sigma=init_sigma,
        max_evals=max_evals,
        tol=tol,
    )

    summary = analyze_results(df)
    summary.to_csv("data/summary.csv", index=False)

    plot_convergence(history_df, "plots")
    plot_summary(summary, "plots")

    print("Eksperyment zakończony. Wyniki i wykresy zapisane.")
