import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# load
df = pd.read_csv("results/results.csv")
df_s = df[df["success"] == True].copy()

print(f"Total rows: {len(df)}")
print(f"Successful trials: {len(df_s)}")
print()

# success rate
print("=== SUCCESS RATE ===")
sr = df.groupby(["algorithm", "env_class"])["success"].mean().unstack()
print(sr.round(3))
print()

# means table
print("=== MEAN METRICS BY ALGORITHM × ENV ===")
means = df_s.groupby(["algorithm", "env_class"])[
    ["path_length", "time_ms", "nodes_explored", "smoothness", "altitude_utilisation"]
].mean()
print(means.round(3))
print()

# two way ANOVA
print("=== TWO-WAY ANOVA ===")
for metric in ["path_length", "time_ms", "nodes_explored", "smoothness", "altitude_utilisation"]:
    model = ols(f'{metric} ~ C(algorithm) * C(env_class)', data=df_s).fit()
    table = sm.stats.anova_lm(model, typ=2)
    print(f"\n--- {metric} ---")
    print(table.round(4))

# cohen's d: A* vs RRT* per env, on path_length and time_ms
print("\n=== COHEN'S D (A* vs RRT*) ===")
for metric in ["path_length", "time_ms"]:
    print(f"\n  {metric}:")
    for env in ["random", "canyon", "clustered", "maze"]:
        g1 = df_s[(df_s.algorithm == "astar")    & (df_s.env_class == env)][metric].dropna()
        g2 = df_s[(df_s.algorithm == "rrt_star") & (df_s.env_class == env)][metric].dropna()
        if len(g1) == 0 or len(g2) == 0:
            print(f"    {env}: no data")
            continue
        pooled_sd = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        d = (g1.mean() - g2.mean()) / pooled_sd if pooled_sd > 0 else 0
        print(f"    {env}: d = {d:.3f}  (A* mean={g1.mean():.2f}, RRT* mean={g2.mean():.2f})")

# tukey HSD: time_ms across env classes
print("\n=== TUKEY HSD: time_ms across env_class ===")
tukey = pairwise_tukeyhsd(df_s["time_ms"], df_s["env_class"])
print(tukey)