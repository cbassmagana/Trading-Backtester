"""
3-Year KNN Strategy:
 - build 3-year PCA embeddings and delta vector,
 - match each multi-embedding to next-year returns,
 - train KNN on rolling 3-year embeddings up to 2015-17,
 - predict 2019 returns for 2016-2018 sequences using weighted KNN,
 - select top-30 predictions for weighted portfolio for 2019,
 - compute realized weighted return for 2019.

 Conceptually this aims to learn which companies are on similar trajectories to other companies that have previously
 experienced strong growth in the following year to come.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pinecone import Pinecone
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


RANDOM_SEED = 17
np.random.seed(RANDOM_SEED)

PINECONE_API_KEY = "pcsk_2vxELT_2ujdmZF6oZKfnusCEdC8dHcZzTP6y4koDZnxR5XmgMo8Fg7aDidwDp9tBdmFHQ3" # "REQUEST FROM AUTHOR"
PQT_PATH = "SourceData/company_info.pqt"

EMBEDDING_DIMENSION = 1536
INDEX_NAME = "10k-filings-index"
NAMESPACE = "large_sample"

REPORTS_YEAR = 2018       # returns for 2019
PCA_DIM = 64
SEQUENCE_YEARS = 3
KNN_NEIGHBORS = 50
TOTAL_INVESTMENT = 1_000_000

OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
existing_names = [idx["name"] for idx in pc.list_indexes()]


def normalize_cik(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = re.sub(r"\.0$", "", s)
    s = s.lstrip("0")
    return s if s else None

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except:
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
            return None
        return (close_next.iloc[-1] / close_year.iloc[-1]) - 1.0
    except:
        return None

def fetch_vectors_by_year(year: int, namespace: str):
    try:
        response = index.query(
            vector=[0] * EMBEDDING_DIMENSION,
            top_k=10000,
            include_metadata=True,
            include_values=True,
            filter={"year": {"$eq": int(year)}},
            namespace=namespace
        )
        return response.get("matches", [])
    except Exception as e:
        print(f"[Warning] Query failed for year {year}: {e}")
        return []

def fetch_all_vectors_for_namespace(namespace, years=range(2010, 2020)):
    all_matches = []
    for year in years:
        print(f"[Info] Fetching vectors for year {year}")
        matches = fetch_vectors_by_year(year, namespace)
        print(f"[Info] {len(matches)} vectors fetched for {year}")
        all_matches.extend(matches)
    print(f"[Info] Total vectors fetched: {len(all_matches)}")
    return all_matches

def apply_pca_to_matches(matches, n_components=PCA_DIM):
    if not matches:
        return matches
    valid = [m for m in matches if m.get("values") is not None]
    embeddings = np.array([m["values"] for m in valid], dtype=float)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(embeddings)
    for i, m in enumerate(valid):
        m["values"] = reduced[i].tolist()
    return matches

def build_3yr_embeddings(df_vectors, comp_df, start_years):
    price_lookup = {
        (row['cik'], row['fyear']): safe_float(row['prcc_c'])
        for _, row in comp_df.iterrows()
        if not pd.isna(row['cik']) and not pd.isna(row['fyear'])
    }

    def compute_next_year_return(cik, end_year):
        p0 = price_lookup.get((cik, end_year), np.nan)
        p1 = price_lookup.get((cik, end_year + 1), np.nan)
        if np.isnan(p0) or np.isnan(p1) or p0 == 0:
            return np.nan
        if (p1 / p0) < 0 or (p1 / p0) > 100:
            return np.nan
        return (p1 / p0) - 1.0

    records = []
    grouped = df_vectors.groupby('cik')
    for cik, group in grouped:
        group = group.set_index('fyear')
        for start in start_years:
            years_needed = list(range(start, start + SEQUENCE_YEARS))
            if not all(y in group.index for y in years_needed):
                continue
            vectors = [group.loc[y, 'vector'] for y in years_needed]
            delta = vectors[-1] - vectors[0]
            embedding = np.concatenate([*vectors, delta])
            next_return = compute_next_year_return(cik, years_needed[-1])
            if np.isnan(next_return):
                continue
            records.append({'cik': cik, 'start_year': start, 'embedding': embedding, 'next_return': next_return})
    return pd.DataFrame(records)


def main():
    print("[Info] Loading company data from Parquet...")
    comp_df = pd.read_parquet(PQT_PATH, columns=["cik", "fyear", "prcc_c"])
    comp_df['cik'] = comp_df['cik'].apply(normalize_cik)
    comp_df['fyear'] = pd.to_numeric(comp_df['fyear'], errors='coerce').astype('Int64')
    comp_df['prcc_c'] = pd.to_numeric(comp_df['prcc_c'], errors='coerce')
    comp_df = comp_df.dropna(subset=['cik', 'fyear'])
    print(f"[Info] Loaded {len(comp_df)} company records after cleaning.")

    years = range(2010, 2020)
    matches = fetch_all_vectors_for_namespace(NAMESPACE, years)
    matches = apply_pca_to_matches(matches, n_components=PCA_DIM)

    recs = []
    for m in matches:
        vec = m.get("values")
        if vec is None:
            continue
        meta = m.get("metadata", {}) or {}
        cik = normalize_cik(meta.get('cik'))
        try:
            fyear = int(float(meta.get('year', np.nan)))
        except:
            continue
        recs.append({'cik': cik, 'fyear': fyear, 'vector': np.array(vec)})

    vectors_df = pd.DataFrame(recs).dropna(subset=['cik', 'fyear'])
    print(f"[Info] Vectors dataframe size: {len(vectors_df)}")

    train_years_seq = range(2010, REPORTS_YEAR - SEQUENCE_YEARS + 1)
    train_df = build_3yr_embeddings(vectors_df, comp_df, train_years_seq)
    print(f"[Info] Training embeddings: {len(train_df)} sequences")

    test_df = build_3yr_embeddings(vectors_df, comp_df, [REPORTS_YEAR - SEQUENCE_YEARS + 1])
    print(f"[Info] Test embeddings: {len(test_df)} sequences")

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError("No valid training/test sequences found.")

    knn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, metric='cosine')
    X_train = np.stack(train_df['embedding'].values)
    y_train = train_df['next_return'].values
    knn.fit(X_train)

    X_test = np.stack(test_df['embedding'].values)
    distances, indices = knn.kneighbors(X_test)
    similarities = 1 - distances
    preds = []
    for i, neighbors_idx in enumerate(indices):
        w = similarities[i]
        r = y_train[neighbors_idx]
        weighted_return = np.sum(w * r) / np.sum(w)
        preds.append(weighted_return)
    test_df['predicted_return'] = preds

    top30 = test_df.nlargest(30, 'predicted_return').copy()
    top30['weight'] = top30['predicted_return'] / top30['predicted_return'].sum()

    price_lookup = {(row['cik'], row['fyear']): safe_float(row['prcc_c']) for _, row in comp_df.iterrows()}
    final_values = []
    for _, row in top30.iterrows():
        p0 = price_lookup.get((row['cik'], REPORTS_YEAR), np.nan)
        p1 = price_lookup.get((row['cik'], REPORTS_YEAR + 1), np.nan)
        final_values.append(0 if np.isnan(p0) or np.isnan(p1) else (p1 / p0) * row['weight'])
    portfolio_return = sum(final_values) - 1

    sp500_return = attempt_fetch_sp500_return(year_end=REPORTS_YEAR)
    avg_return = test_df['next_return'].mean()
    print(f"[Result] Portfolio return 2019: {portfolio_return:.2%}")
    print(f"[Benchmark] S&P500 2019 return: {sp500_return:.2%}")
    print(f"[Benchmark] Avg dataset return: {avg_return:.2%}")

    top30['actual_return'] = top30['next_return']

    existing = [
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("experiment_results_") and f.endswith(".json") and f.split("_")[-1].split(".")[0].isdigit()
    ]
    next_index = max(existing) + 1 if existing else 1
    output_path = os.path.join(OUTPUT_DIR, f"experiment_results_{next_index}.json")
    experiment_data = {
        "parameters": {
            "namespace": NAMESPACE,
            "years": list(years),
            "pca_dim": PCA_DIM,
            "reports_year": REPORTS_YEAR,
            "random_seed": RANDOM_SEED,
            "model_type": "KNN",
            "n_neighbors": KNN_NEIGHBORS
        },
        "results": {
            "portfolio_return": float(portfolio_return),
            "sp500_return": float(sp500_return),
            "avg_dataset_return": float(avg_return),
            "num_train": int(len(train_df)),
            "num_test": int(len(test_df))
        },
        "top30_holdings": top30[['cik', 'start_year', 'predicted_return', 'actual_return', 'weight']].to_dict(orient='records'),
        "comments": "3-year KNN approach with delta vector appended"
    }
    with open(output_path, "w") as f:
        json.dump(experiment_data, f, indent=4)
    print(f"[Saved] Experiment results written to {output_path}")


if __name__ == "__main__":
    main()
