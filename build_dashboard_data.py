"""
Québec Urban Trees K-NN Dashboard
Génère dashboard_data.json depuis l'API CKAN de Données Québec.

Exécution : python build_dashboard_data.py
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CKAN_URL = "https://www.donneesquebec.ca/recherche/api/3/action/datastore_search"
RESOURCE_ID = "13a51853-a5b5-4add-8791-02ccba5c1be7"
OUTPUT_FILE = Path("dashboard_data.json")
MAX_TREES_FOR_DASHBOARD = 7000
MAX_NEIGHBORS = 30
RANDOM_STATE = 42


def fetch_ckan_records(resource_id: str, limit: int = 5000) -> pd.DataFrame:
    records, offset, total = [], 0, None
    while True:
        params = {"resource_id": resource_id, "limit": limit, "offset": offset}
        response = requests.get(CKAN_URL, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"Réponse CKAN non valide : {payload}")
        result = payload["result"]
        batch = result.get("records", [])
        total = result.get("total", len(batch))
        records.extend(batch)
        print(f"[API] Lignes chargées : {len(records)} / {total}")
        if not batch or len(records) >= total:
            break
        offset += limit
    return pd.DataFrame(records)


def clean_tree_data(df: pd.DataFrame) -> pd.DataFrame:
    expected = ["ID", "TYPE_LIEU", "NOM_LATIN", "NOM_FRANCAIS", "TYPE_ARBRE", "DIAMETRE", "POSITION_MESURE", "MULTI_TRONC", "DATE_PLANTE", "TYPE_PROP", "LATITUDE", "LONGITUDE", "NOM_TOPOGRAPHIE"]
    df = df[[c for c in expected if c in df.columns]].copy()

    for col in ["TYPE_LIEU", "NOM_LATIN", "NOM_FRANCAIS", "TYPE_ARBRE", "POSITION_MESURE", "MULTI_TRONC", "TYPE_PROP", "NOM_TOPOGRAPHIE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "None": np.nan, "nan": np.nan})

    for col in ["DIAMETRE", "LATITUDE", "LONGITUDE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "DATE_PLANTE" in df.columns:
        df["DATE_PLANTE"] = pd.to_datetime(df["DATE_PLANTE"], errors="coerce")
        df["ANNEE_PLANTE"] = df["DATE_PLANTE"].dt.year
        df["AGE_ESTIME"] = datetime.now().year - df["ANNEE_PLANTE"]
        df.loc[(df["AGE_ESTIME"] < 0) | (df["AGE_ESTIME"] > 200), "AGE_ESTIME"] = np.nan

    df = df.dropna(subset=["ID", "TYPE_ARBRE", "DIAMETRE", "LATITUDE", "LONGITUDE"])
    df = df[df["LATITUDE"].between(46.6, 47.1) & df["LONGITUDE"].between(-71.7, -70.9)]
    df = df[df["TYPE_ARBRE"].isin(["Feuillu", "Conifère"])].copy()
    df["ID"] = df["ID"].astype(str)
    if "NOM_TOPOGRAPHIE" in df.columns:
        df["SECTEUR_AFFICHE"] = df["NOM_TOPOGRAPHIE"].fillna("Secteur non précisé")
    return df.drop_duplicates(subset=["ID"]).reset_index(drop=True)


def sample_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MAX_TREES_FOR_DASHBOARD:
        return df.copy()
    parts = []
    for _, g in df.groupby("TYPE_ARBRE"):
        n = min(len(g), max(100, int(MAX_TREES_FOR_DASHBOARD * len(g) / len(df))))
        parts.append(g.sample(n=n, random_state=RANDOM_STATE))
    out = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return out.head(MAX_TREES_FOR_DASHBOARD).copy()


def get_features(df: pd.DataFrame, include_target: bool = False):
    numeric = [c for c in ["LATITUDE", "LONGITUDE", "DIAMETRE", "AGE_ESTIME"] if c in df.columns]
    categorical = [c for c in ["TYPE_LIEU", "TYPE_PROP", "POSITION_MESURE", "MULTI_TRONC"] if c in df.columns]
    if include_target and "TYPE_ARBRE" in df.columns:
        categorical.append("TYPE_ARBRE")
    return numeric, categorical


def preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def train_knn_classifier(df: pd.DataFrame) -> dict:
    numeric, categorical = get_features(df, include_target=False)
    features = numeric + categorical
    X, y = df[features], df["TYPE_ARBRE"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=RANDOM_STATE, stratify=y)
    results, best, best_model = [], None, None
    for k in [3, 5, 7, 11, 15, 21]:
        model = Pipeline([("preprocess", preprocessor(numeric, categorical)), ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance"))])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        row = {
            "k": k,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
            "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
        }
        results.append(row)
        if best is None or row["f1_macro"] > best["f1_macro"]:
            best, best_model = row, model
    pred = best_model.predict(X_test)
    labels = sorted(y.unique().tolist())
    return {"task": "Classification K-NN du type d'arbre", "target": "TYPE_ARBRE", "features": features, "best_k": best["k"], "selection_metric": "f1_macro", "model_results": results, "labels": labels, "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(), "test_size": len(y_test)}


def compute_neighbors(df: pd.DataFrame) -> dict:
    numeric, categorical = get_features(df, include_target=True)
    features = numeric + categorical
    X = preprocessor(numeric, categorical).fit_transform(df[features])
    nn = NearestNeighbors(n_neighbors=min(MAX_NEIGHBORS + 1, len(df)), metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    ids = df["ID"].tolist()
    graph = {}
    for i, tree_id in enumerate(ids):
        arr = []
        for dist, j in zip(distances[i], indices[i]):
            if ids[j] == tree_id:
                continue
            arr.append({"id": ids[j], "distance": round(float(dist), 4)})
            if len(arr) >= MAX_NEIGHBORS:
                break
        graph[tree_id] = arr
    return graph


def json_records(df: pd.DataFrame) -> list[dict]:
    cols = ["ID", "TYPE_LIEU", "NOM_LATIN", "NOM_FRANCAIS", "TYPE_ARBRE", "DIAMETRE", "POSITION_MESURE", "MULTI_TRONC", "TYPE_PROP", "LATITUDE", "LONGITUDE", "ANNEE_PLANTE", "AGE_ESTIME", "SECTEUR_AFFICHE"]
    cols = [c for c in cols if c in df.columns]
    records = []
    for rec in df[cols].to_dict(orient="records"):
        records.append({k: (None if pd.isna(v) else v) for k, v in rec.items()})
    return records


def payload(df: pd.DataFrame, metrics: dict, neighbors: dict) -> dict:
    return {
        "metadata": {"project": "Québec Urban Trees K-NN Dashboard", "generated_at": datetime.now(timezone.utc).isoformat(), "source": "Données Québec — Arbres répertoriés — Ville de Québec", "resource_id": RESOURCE_ID, "api_url": CKAN_URL, "license_data": "CC-BY 4.0", "rows_in_dashboard": len(df), "note": "NOM_TOPOGRAPHIE n'est pas exposé directement dans le JSON final."},
        "summary": {"type_distribution": df["TYPE_ARBRE"].value_counts().reset_index().rename(columns={"TYPE_ARBRE": "type_arbre", "count": "count"}).to_dict(orient="records"), "top_species": df["NOM_FRANCAIS"].fillna("Non précisé").value_counts().head(15).reset_index().rename(columns={"NOM_FRANCAIS": "nom_francais", "count": "count"}).to_dict(orient="records"), "diameter_stats": {"min": round(float(df["DIAMETRE"].min()), 2), "max": round(float(df["DIAMETRE"].max()), 2), "mean": round(float(df["DIAMETRE"].mean()), 2), "median": round(float(df["DIAMETRE"].median()), 2)}},
        "filters": {"type_arbre": sorted(df["TYPE_ARBRE"].dropna().unique().tolist()), "type_lieu": sorted(df.get("TYPE_LIEU", pd.Series(dtype=str)).dropna().unique().tolist()), "type_prop": sorted(df.get("TYPE_PROP", pd.Series(dtype=str)).dropna().unique().tolist()), "nom_francais": sorted(df["NOM_FRANCAIS"].dropna().unique().tolist())[:500]},
        "model": metrics,
        "trees": json_records(df),
        "neighbors": neighbors,
    }


def main():
    raw = fetch_ckan_records(RESOURCE_ID)
    print(f"[build] Lignes brutes : {len(raw):,}")
    clean = clean_tree_data(raw)
    print(f"[build] Lignes nettoyées : {len(clean):,}")
    dash = sample_for_dashboard(clean)
    print(f"[build] Lignes dashboard : {len(dash):,}")
    metrics = train_knn_classifier(clean)
    neighbors = compute_neighbors(dash)
    OUTPUT_FILE.write_text(json.dumps(payload(dash, metrics, neighbors), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] Fichier généré : {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
