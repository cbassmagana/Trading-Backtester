"""
Outlier-based mean-reversion strategy:
 - train per-year regression on (2010-2017) embeddings -> same-year returns,
 - compute average text-embeddings per company over 2010-2017,
 - identify embedding outliers in 2018 based on distance from company average,
 - compute expected same-year return difference from average using regression,
 - select top/bottom 30 outliers by expected difference for 2019 returns,
 - compare performance of top 30 to bottom 30 to SP500.

 Conceptually aims to find companies that have abnormal reporting and public sentiment compared to historical average,
 and anticipates mean reversion of price the following year.
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
PQT_PATH = "SourceData/company_info.pqt"

EMBEDDING_DIMENSION = 1536
INDEX_NAME = "10k-filings-index"
NAMESPACE = "large_sample"

REPORTS_YEAR = 2016
PCA_DIM = 64

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

def attempt_fetch_sp500_return(year_end=REPORTS_YEAR, csv_path="SourceData/sp500_data.csv"):
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found at: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
        close_year = df[df["Date"].dt.year == year_end]["Close"]
        close_next = df[df["Date"].dt.year == (year_end + 1)]["Close"]
        if close_year.empty or close_next.empty:
            print(f"[Warning] Missing data for year(s) {year_end} or {year_end+1}")
            return None
        return (close_next.iloc[-1] / close_year.iloc[-1]) - 1.0
    except Exception as e:
        print(f"[Error] Failed to compute S&P 500 return: {e}")
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
        all_matches.extend(response.get("matches", []))
    except Exception as e:
        print(f"[Warning] Query failed for year {year}: {e}")
    return all_matches

def fetch_all_vectors(namespace, years=range(2010, 2020)):
    all_matches = []
    for year in years:
        print(f"[Info] Fetching vectors for year {year}")
        matches = fetch_vectors_by_year(year, namespace)
        print(f"[Info] {len(matches)} vectors fetched for {year}")
        all_matches.extend(matches)
    print(f"[Info] Total vectors fetched: {len(all_matches)}")
    return all_matches

def apply_pca(matches, n_components=PCA_DIM):
    valid = [m for m in matches if m.get("values") is not None]
    if not valid:
        return matches
    embeddings = np.array([m["values"] for m in valid], dtype=float)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(embeddings)
    for i, m in enumerate(valid):
        m["values"] = reduced[i].tolist()
    print(f"[Info] PCA applied: 1536 -> {n_components}")
    return matches

def compute_next_year_return(row, price_lookup):
    cik = normalize_cik(row["cik"])
    fyear = int(row["fyear"])
    p0 = price_lookup.get((cik, fyear), np.nan)
    p1 = price_lookup.get((cik, fyear + 1), np.nan)
    if np.isnan(p0) or np.isnan(p1) or p0 == 0:
        return np.nan
    if (p1 / p0) < 0 or (p1 / p0) > 100:
        return np.nan
    return (p1 / p0) - 1.0

def compute_same_year_return(row, price_lookup):
    cik = normalize_cik(row["cik"])
    fyear = int(row["fyear"])
    p0 = price_lookup.get((cik, fyear-1), np.nan)
    p1 = price_lookup.get((cik, fyear), np.nan)
    if np.isnan(p0) or np.isnan(p1) or p0 == 0:
        return np.nan
    if (p1 / p0) < 0 or (p1 / p0) > 100:
        return np.nan
    return (p1 / p0) - 1.0


def main():
    print("[Info] Loading company data...")
    comp_df = pd.read_parquet(PQT_PATH, columns=["cik","fyear","prcc_c"])
    comp_df["cik"] = comp_df["cik"].apply(normalize_cik)
    comp_df["fyear"] = pd.to_numeric(comp_df["fyear"], errors="coerce").astype("Int64")
    comp_df["prcc_c"] = pd.to_numeric(comp_df["prcc_c"], errors="coerce")
    comp_df = comp_df.dropna(subset=["cik","fyear"])
    print(f"[Info] {len(comp_df)} records loaded.")

    matches = fetch_all_vectors(NAMESPACE, years=range(2010,2020))
    matches = apply_pca(matches, n_components=PCA_DIM)

    recs = []
    for m in matches:
        metadata = m.get("metadata", {}) or {}
        vec = m.get("values")
        if vec is None:
            continue
        cik = normalize_cik(metadata.get("cik"))
        fyear = int(metadata.get("year", np.nan))
        recs.append({"cik":cik, "fyear":fyear, "vector":np.array(vec,dtype=float)})

    vectors_df = pd.DataFrame(recs).dropna(subset=["cik","fyear"])
    vectors_df["fyear"] = vectors_df["fyear"].astype(int)
    merged = vectors_df.merge(comp_df, how="left", on=["cik","fyear"])
    print(f"[Info] Merged vectors with prices: {merged['prcc_c'].notna().sum()} rows have price info.")

    price_lookup = {(normalize_cik(r["cik"]), int(r["fyear"])): safe_float(r["prcc_c"]) for _,r in comp_df.iterrows()}
    merged["same_return"] = merged.apply(lambda r: compute_same_year_return(r, price_lookup), axis=1)
    merged["next_return"] = merged.apply(lambda r: compute_next_year_return(r, price_lookup), axis=1)
    print(f"[Info] Same-year and next-year returns computed: {merged['same_return'].notna().sum()} valid")

    train_df = merged[(merged["fyear"]>=2010) & (merged["fyear"] <= REPORTS_YEAR - 1) & merged["same_return"].notna()].copy()
    X_train = np.stack(train_df["vector"].values)
    y_train = train_df["same_return"].values

    model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"[Info] Regression trained on {len(y_train)} samples.")

    avg_embeds = merged[(merged["fyear"]>=2010) & (merged["fyear"] <= REPORTS_YEAR - 1)].groupby("cik")["vector"].apply(lambda arrs: np.mean(np.stack(arrs.values), axis=0))

    valid_ciks = set(cik for (cik, fyear) in price_lookup.keys() if fyear in [REPORTS_YEAR - 1, REPORTS_YEAR, REPORTS_YEAR + 1])
    df_year_end = merged[(merged["fyear"] == REPORTS_YEAR) & (merged["cik"].isin(valid_ciks))].copy()

    df_year_end["avg_vector"] = df_year_end["cik"].map(avg_embeds)
    df_year_end = df_year_end.dropna(subset=["avg_vector"])
    df_year_end = df_year_end[df_year_end["same_return"].notna() & df_year_end["next_return"].notna()].copy()
    df_year_end["distance"] = df_year_end.apply(lambda r: np.linalg.norm(r["vector"]-r["avg_vector"]), axis=1)
    df_year_end_200 = df_year_end.nlargest(200,"distance")

    def safe_predict(vec):
        try:
            return float(model.predict(np.asarray(vec).reshape(1, -1))[0])
        except Exception:
            return np.nan

    df_year_end_200["pred_new"] = df_year_end_200["vector"].apply(safe_predict)
    df_year_end_200["pred_avg"] = df_year_end_200["avg_vector"].apply(safe_predict)
    df_year_end_200["pred_delta"] = df_year_end_200["pred_new"] - df_year_end_200["pred_avg"]

    top30 = df_year_end_200.nlargest(30, "pred_delta").copy()
    bottom30 = df_year_end_200.nsmallest(30, "pred_delta").copy()

    top30["actual_return"] = top30["next_return"]
    bottom30["actual_return"] = bottom30["next_return"]
    portfolio_return_top = top30["actual_return"].mean()
    portfolio_return_bottom = bottom30["actual_return"].mean()

    sp500_return = attempt_fetch_sp500_return(year_end=REPORTS_YEAR)
    avg_return = merged[merged["fyear"] == REPORTS_YEAR]["next_return"].mean()

    print(f"[Result] Outlier portfolio 2019 return (top 30): {portfolio_return_top:.2%}")
    print(f"[Result] Outlier portfolio 2019 return (bottom 30): {portfolio_return_bottom:.2%}")
    print(f"[Benchmark] S&P500 2019 return: {sp500_return:.2%}")
    print(f"[Benchmark] Avg dataset return: {avg_return:.2%}")

    existing = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(OUTPUT_DIR) if f.startswith("experiment_results_") and f.endswith(".json") and f.split("_")[-1].split(".")[0].isdigit()]
    next_index = max(existing)+1 if existing else 1
    output_path = os.path.join(OUTPUT_DIR,f"experiment_results_{next_index}.json")

    experiment_data = {
        "parameters":{
            "namespace":NAMESPACE,
            "pca_dim":PCA_DIM,
            "reports_year": REPORTS_YEAR,
            "random_seed":RANDOM_SEED,
            "model_type":"RandomForestRegressor",
            "n_estimators":50
        },
        "results":{
            "portfolio_return_top":float(portfolio_return_top),
            "portfolio_return_bottom":float(portfolio_return_bottom),
            "sp500_return":float(sp500_return),
            "avg_dataset_return":float(avg_return),
            "num_train_rows": int(len(train_df)),
            "num_test_candidates": int(len(df_year_end))
        },
        "top_holdings":top30[["cik","fyear","distance","pred_delta","actual_return"]].to_dict(orient="records"),
        "bottom_holdings":bottom30[["cik","fyear","distance","pred_delta","actual_return"]].to_dict(orient="records"),
        "comments": ""
    }

    with open(output_path,"w") as f:
        json.dump(experiment_data,f,indent=4)
    print(f"[Saved] Experiment results written to {output_path}.")


if __name__=="__main__":
    main()
