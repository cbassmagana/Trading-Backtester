"""
Standard Regression:
 - compute next-year returns for each vector,
 - train a regressor on embeddings through 2017 (example),
 - predict for 2018 embeddings (2019 returns),
 - select top-30 predicted to construct portfolio,
 - compute realized portfolio return for 2019.

 Conceptually this aims to learn which types of company reports for year X correspond to strong returns in year X+1.
"""

import os
from pinecone import Pinecone
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import json


RANDOM_SEED = 17
np.random.seed(RANDOM_SEED)

PINECONE_API_KEY = "REQUEST FROM AUTHOR"
PQT_PATH = "SourceData/comp_total_q.pqt"

EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 100
INDEX_NAME = "10k-filings-index"
NAMESPACE = "large_sample"

PCA_DIM = 64
REPORTS_YEAR = 2018
TOTAL_INVESTMENT = 1_000_000

OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
existing_names = [idx["name"] for idx in pc.list_indexes()]


def normalize_str(x):
    if pd.isna(x):
        return None
    return str(x).strip()

def normalize_cik(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = re.sub(r"\.0$", "", s)
    s = s.lstrip("0")
    if s == "":
        return None
    return s

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def attempt_fetch_sp500_return(year_end=REPORTS_YEAR, csv_path="Source_Data/sp500_data.csv"):
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.sort_values("Date")
        close_year = df[df["Date"].dt.year == year_end]["Close"]
        close_next = df[df["Date"].dt.year == (year_end + 1)]["Close"]
        if close_year.empty or close_next.empty:
            print(f"[Warning] Missing data for year(s) {year_end} or {year_end + 1}")
            return None
        close_year_last = close_year.iloc[-1]
        close_next_last = close_next.iloc[-1]
        return_pct = (close_next_last / close_year_last) - 1.0
        return return_pct
    except Exception as e:
        print(f"[Error] Failed to compute S&P 500 return from CSV: {e}")
        return None

def fetch_vectors_by_year(year: int, namespace: str):
    all_matches = []
    try:
        response = index.query(
            vector=[0] * EMBEDDING_DIMENSION,
            top_k=10000,
            include_metadata=True,
            include_values=True,
            filter={"year": {"$eq": int(year)}},
            namespace=namespace
        )
        matches = response.get("matches", [])
        all_matches.extend(matches)
    except Exception as e:
        print(f"[Warning] Query failed for year {year}: {e}")
    return all_matches

def fetch_all_vectors_for_namespace(namespace, years=range(2010, 2020)):
    all_matches = []
    for year in years:
        print(year)
        try:
            matches = fetch_vectors_by_year(year, namespace)
        except Exception as e:
            print(f"[Warning] fetch_vectors_by_year failed for year {year}: {e}")
            matches = []
        if matches:
            print(len(matches))
            all_matches.extend(matches)
    print(f"Fetched total vectors across years {min(years)}-{max(years)}: {len(all_matches)}")
    return all_matches

def apply_pca_to_matches(matches, n_components=PCA_DIM):
    if not matches:
        print("[Warning] No matches available for PCA reduction.")
        return matches
    print(f"[Info] Applying PCA reduction ({len(matches)} vectors, 1536 -> {n_components})...")
    valid = [m for m in matches if m.get("values") is not None]
    embeddings = np.array([m["values"] for m in valid], dtype=float)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(embeddings)
    for i, m in enumerate(valid):
        m["values"] = reduced[i].tolist()
    print(f"[Info] PCA reduction complete. Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
    return matches


def main():
    print("[Info] Loading company data from Parquet...")
    comp_df = pd.read_parquet(PQT_PATH, columns=["cik", "fyear", "prcc_c"])
    comp_df["cik"] = comp_df["cik"].apply(normalize_cik)
    comp_df["fyear"] = pd.to_numeric(comp_df["fyear"], errors="coerce").astype("Int64")
    comp_df["prcc_c"] = pd.to_numeric(comp_df["prcc_c"], errors="coerce")
    comp_df = comp_df.dropna(subset=["cik", "fyear"])
    print(f"[Info] Loaded {len(comp_df)} company records after cleaning.")

    years = list(range(2010, 2020))
    matches = fetch_all_vectors_for_namespace(NAMESPACE, years=years)
    matches = apply_pca_to_matches(matches, n_components=PCA_DIM)

    recs = []
    for m in matches:
        metadata = m.get("metadata", {}) or {}
        vec = m.get("values")
        if vec is None:
            continue

        cik = None
        fyear = None
        for k in ["cik"]:
            if k in metadata:
                cik = normalize_cik(metadata[k])
                break
        for k in ["year"]:
            if k in metadata:
                try:
                    fyear = int(float(metadata[k]))
                except Exception:
                    fyear = None
                break

        recs.append({
            "cik": cik,
            "fyear": fyear,
            "vector": np.array(vec, dtype=float)
        })

    vectors_df = pd.DataFrame(recs)
    vectors_df = vectors_df.dropna(subset=["cik", "fyear"])
    vectors_df["fyear"] = vectors_df["fyear"].astype(int)
    print(f"[Info] Built vectors dataframe with {len(vectors_df)} rows")

    merged = vectors_df.merge(
        comp_df,
        how="left",
        on=["cik", "fyear"],
        suffixes=("", "_comp")
    )

    print(f"[Info] After merging, {merged['prcc_c'].notna().sum()} vectors have stock price info available.")

    price_lookup = {
        (normalize_cik(row["cik"]), int(row["fyear"])): safe_float(row["prcc_c"])
        for _, row in comp_df.iterrows()
        if not pd.isna(row["fyear"]) and not pd.isna(row["cik"])
    }

    def compute_next_year_return(row):
        cik = normalize_cik(row["cik"])
        fyear = int(row["fyear"])
        p0 = price_lookup.get((cik, fyear), np.nan)
        p1 = price_lookup.get((cik, fyear + 1), np.nan)
        if np.isnan(p0) or np.isnan(p1) or p0 == 0:
            return np.nan
        if (p1 / p0) < 0 or (p1 / p0) > 100:
            return np.nan
        return (p1 / p0) - 1.0

    merged["next_return"] = merged.apply(compute_next_year_return, axis=1)
    available_count = merged["next_return"].notna().sum()
    print(f"[Info] Computed next-year returns: {available_count} valid rows found.")

    model_df = merged.dropna(subset=["next_return"]).copy()
    print(f"[Info] Model dataset size: {len(model_df)}")

    train_mask = model_df["fyear"] <= REPORTS_YEAR - 1
    test_mask = model_df["fyear"] == REPORTS_YEAR

    X_train = np.stack(model_df.loc[train_mask, "vector"].values) if train_mask.any() else np.empty((0, PCA_DIM))
    y_train = model_df.loc[train_mask, "next_return"].values if train_mask.any() else np.empty((0,))
    X_test = np.stack(model_df.loc[test_mask, "vector"].values) if test_mask.any() else np.empty((0, PCA_DIM))
    y_test = model_df.loc[test_mask, "next_return"].values if test_mask.any() else np.empty((0,))

    print(f"[Info] Training samples: {len(y_train)}, Testing (2018): {len(y_test)}")

    if len(y_train) == 0:
        raise RuntimeError("No training data found. Check CIK normalization or year filters.")

    model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    model_df.loc[test_mask, "predicted_return"] = y_pred

    test_df = model_df.loc[test_mask].dropna(subset=["predicted_return"])
    top30 = test_df.nlargest(30, "predicted_return").copy()

    top30["weight"] = top30["predicted_return"] / top30["predicted_return"].sum()
    top30["actual_return"] = top30["next_return"]

    total_investment = TOTAL_INVESTMENT
    top30["investment"] = top30["weight"] * total_investment
    top30["final_value"] = top30["investment"] * (1 + top30["actual_return"])
    portfolio_return = (top30["final_value"].sum() / total_investment) - 1

    print(f"[Result] Portfolio return for 2019 (predicted using 2018 reports): {portfolio_return:.2%}")

    sp500_return = attempt_fetch_sp500_return(year_end=REPORTS_YEAR)
    avg_return = test_df["next_return"].mean()

    print(f"[Benchmark] S&P500 2019 return: {sp500_return:.2%}")
    print(f"[Benchmark] Average dataset return: {avg_return:.2%}")

    existing = [
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir("Outputs")
        if f.startswith("experiment_results_") and f.endswith(".json") and f.split("_")[-1].split(".")[0].isdigit()
    ]
    next_index = max(existing) + 1 if existing else 1
    output_path = os.path.join("Outputs", f"experiment_results_{next_index}.json")

    experiment_data = {
        "parameters": {
            "namespace": NAMESPACE,
            "years": years,
            "pca_dim": PCA_DIM,
            "reports_year": REPORTS_YEAR,
            "random_seed": RANDOM_SEED,
            "model_type": "RandomForestRegressor",
            "n_estimators": 50
        },
        "results": {
            "portfolio_return": float(portfolio_return),
            "sp500_return": float(sp500_return),
            "avg_dataset_return": float(avg_return),
            "num_train": int(len(y_train)),
            "num_test": int(len(y_test))
        },
        "top30_holdings": top30[["cik", "fyear", "predicted_return", "actual_return", "weight"]].to_dict(orient="records"),
        "comments": "Add your notes or interpretation of results here."
    }

    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=4)

    print(f"[Saved] Experiment results written to {output_path}")


if __name__ == "__main__":
    main()