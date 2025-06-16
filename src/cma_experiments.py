import argparse
import time

import cma
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# benchmark functions
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


# cmaes wrapper
def run_cma_es(
    func, dim, init_sigma, init_ps_random, base_seed, generator_name, max_evals, tol
):
    es_rng = {"seed": base_seed, "verb_log": 0, "verb_disp": 0}
    x0 = np.zeros(dim)
    es = cma.CMAEvolutionStrategy(x0, init_sigma, es_rng)
    # RNG for random initialization of p_sigma
    if init_ps_random:
        if generator_name == "mersenne":
            rng = np.random.RandomState(base_seed)
        else:  # 'pcg'
            rng = np.random.Generator(np.random.PCG64(base_seed))
        es.ps = rng.standard_normal(dim)

    history = []
    evals = 0
    while not es.stop() and evals < max_evals:
        solutions = es.ask()
        fitnesses = [func(x) for x in solutions]
        evals += len(solutions)
        es.tell(solutions, fitnesses)
        best = min(fitnesses)
        history.append((evals, best))
        if best <= tol:
            break
    return {
        "best_fitness": es.best.f,
        "evals": evals,
        "converged": es.best.f <= tol,
        "history": history,
    }


# run function
def run_experiments(output_csv, functions, dims, num_runs, init_sigma, max_evals, tol):
    records = []
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
                            base_seed=seed,
                            generator_name=generator_name,
                            max_evals=max_evals,
                            tol=tol,
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
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    return df


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
                stat, p = wilcoxon(pivot["standard"], pivot["random"])
                summary.append(
                    {
                        "function": func_name,
                        "generator": gen_name,
                        "dim": dim,
                        "stat": stat,
                        "p_value": p,
                        "significant": p < alpha,
                    }
                )
    return pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMA-ES experiment containerized")
    parser.add_argument("runs", type=int, help="Number of runs per combination")
    args = parser.parse_args()

    dims = [2, 10, 30]
    init_sigma = 0.3
    max_evals = 10000
    tol = 1e-8

    df = run_experiments(
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
