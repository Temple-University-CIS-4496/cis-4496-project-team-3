# this script reads scene_table.csv from each movie folder inside a base directory,
# then:
# 1. makes timeline plots for tension, irreversibility, and resolution progress
# 2. computes per-movie distribution stats (mean, std, min, max)
# 3. combines all movie stats into one csv
# 4. clusters movies using resampled tension-over-time sequences
# 5. saves all results into an analysis_results folder

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================================================
# set this path
# this folder should contain the 10 movie folders
# each movie folder should contain scene_table.csv
# =========================================================
BASE_DIR = Path(r"C:\Users\vishr\Downloads\macro_per_script")  # change this
OUTPUT_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# columns to use
# =========================================================
VALUE_COLS = [
    "tension_level",
    "stakes_level",
    "irreversibility_level",
    "uncertainty_level",
    "narrative_velocity",
    "dependency_on_prior_scenes",
    "resolution_progress",
]

PLOT_COLS = [
    "tension_level",
    "irreversibility_level",
    "resolution_progress",
]

TARGET_LEN = 30
N_CLUSTERS = 4

# =========================================================
# helpers
# =========================================================
def get_movie_folders(base_dir: Path):
    return [p for p in base_dir.iterdir() if p.is_dir()]

def load_scene_table(movie_dir: Path):
    csv_path = movie_dir / "scene_table.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)

def make_distribution_stats(df: pd.DataFrame, movie_name: str):
    rows = []
    for col in VALUE_COLS:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "movie": movie_name,
            "variable": col,
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)

def plot_timelines(df: pd.DataFrame, movie_name: str, out_dir: Path):
    x = df["scene_idx"]

    for col in PLOT_COLS:
        if col not in df.columns:
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(x, df[col], marker="o")
        plt.title(f"{movie_name}: {col} by scene")
        plt.xlabel("scene index")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(out_dir / f"{movie_name}_{col}.png", dpi=150)
        plt.close()

def build_tension_sequence(df: pd.DataFrame, target_len=30):
    s = pd.to_numeric(df["tension_level"], errors="coerce").dropna().values.astype(float)
    if len(s) < 2:
        return None
    return resample(s, target_len)

# =========================================================
# main analysis
# =========================================================
def main():
    all_stats = []
    tension_rows = []
    movie_names = []

    movie_folders = get_movie_folders(BASE_DIR)

    for movie_dir in movie_folders:
        movie_name = movie_dir.name
        df = load_scene_table(movie_dir)

        if df is None:
            print(f"skipping {movie_name}: no scene_table.csv")
            continue

        needed = ["scene_idx"] + VALUE_COLS + ["major_turning_point"]
        existing = [c for c in needed if c in df.columns]
        df = df[existing].copy()

        # save cleaned version
        df.to_csv(OUTPUT_DIR / f"{movie_name}_cleaned_scene_table.csv", index=False)

        # make plots
        plot_timelines(df, movie_name, OUTPUT_DIR)

        # distribution stats
        stats_df = make_distribution_stats(df, movie_name)
        if not stats_df.empty:
            stats_df.to_csv(OUTPUT_DIR / f"{movie_name}_distribution_stats.csv", index=False)
            all_stats.append(stats_df)

        # tension sequence for clustering
        if "tension_level" in df.columns:
            tension_seq = build_tension_sequence(df, target_len=TARGET_LEN)
            if tension_seq is not None:
                tension_rows.append(tension_seq)
                movie_names.append(movie_name)

    # combine all movie stats
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
        combined_stats.to_csv(OUTPUT_DIR / "all_movies_distribution_stats.csv", index=False)

    # =========================================================
    # tension clustering
    # =========================================================
    if len(tension_rows) >= N_CLUSTERS:
        X = np.array(tension_rows)
        X_scaled = StandardScaler().fit_transform(X)

        model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        cluster_df = pd.DataFrame({
            "movie": movie_names,
            "cluster": labels
        })
        cluster_df.to_csv(OUTPUT_DIR / "tension_clusters.csv", index=False)

        # plot each cluster's trajectories
        for cluster_id in sorted(cluster_df["cluster"].unique()):
            idx = labels == cluster_id
            members = X[idx]

            plt.figure(figsize=(8, 4))
            for row in members:
                plt.plot(row, alpha=0.35)
            plt.plot(members.mean(axis=0), linewidth=3)
            plt.title(f"cluster {cluster_id}: tension trajectories")
            plt.xlabel("resampled scene position")
            plt.ylabel("tension")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"cluster_{cluster_id}_tension.png", dpi=150)
            plt.close()

        print("tension clustering done")
    else:
        print("not enough movies for clustering")

    print("done")
    print(f"results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()