"""Part B, tasks 1-3 -- RFM table, K-means model selection, segment profiles and PCA.

Run: python -m copilot.part_b_segments      (writes outputs/customer_segments.csv)
Only completed orders describe customer behaviour.
"""
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from copilot.paths import CLEAN, OUT, SEED, Report, ensure_dirs

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
RFM = ["recency", "frequency", "monetary"]


def load_completed():
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
    return orders[orders["status"] == "completed"]


def build_rfm(completed):
    g = completed.groupby("customer_id")
    rfm = pd.DataFrame({
        "recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
        "frequency": g["order_id"].nunique(),
        "monetary": g["order_amount"].sum().round(2),
    }).reset_index()
    return rfm


def name_segments(profile):
    """Business names are assigned from each cluster's profile, not from its arbitrary cluster number."""
    names = {}
    premium = profile["monetary"].idxmax()
    names[premium] = ("Premium Big Spenders",
                      "Protect with a premium desk, early access to electronics launches and a personal "
                      "win-back call as soon as recency rises.")
    rest = profile.drop(index=premium)
    lapsing = rest["recency"].idxmax()
    names[lapsing] = ("Lapsing Occasional Buyers",
                      "Run a low-cost win-back offer (free delivery on the next order) and stop spending "
                      "on those who still do not return.")
    rest = rest.drop(index=lapsing)
    loyal = rest["frequency"].idxmax()
    names[loyal] = ("Loyal Daily Shoppers",
                    "Reward habit with a grocery subscription or loyalty points; cross-sell personal care "
                    "and snacks to raise basket value.")
    for seg in rest.drop(index=loyal).index:
        names[seg] = ("Occasional Light Shoppers",
                      "Grow basket size and frequency with bundled staples offers and a 'third order this "
                      "month' coupon.")
    return names


def main():
    ensure_dirs()
    rep = Report("part_b_segments")
    rfm = build_rfm(load_completed())

    rep("## Task 1 - RFM table (snapshot 2025-10-01 00:00:00, completed orders only)")
    rep(f"customers with at least one completed order: {len(rfm):,}")
    rep(rfm[RFM].describe().round(2).to_string())
    rep(f"recency median {rfm.recency.median():g}, min {rfm.recency.min()}, max {rfm.recency.max()}; "
        f"frequency mean {rfm.frequency.mean():.2f}, max {rfm.frequency.max()}; "
        f"monetary mean {rfm.monetary.mean():,.2f}, max {rfm.monetary.max():,.2f}")

    X = StandardScaler().fit_transform(rfm[RFM])
    rep("\n## Task 2 - K-means on standardised RFM (n_init=10, random_state=42)")
    rep(f"{'K':>3}{'inertia':>12}{'silhouette':>12}")
    models = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        models[k] = km
        rep(f"{k:>3}{km.inertia_:>12.1f}{silhouette_score(X, km.labels_):>12.3f}")
    rep(f"chosen K = {FINAL_K}")

    km = models[FINAL_K]
    rfm["segment_id"] = km.labels_
    profile = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                            frequency=("frequency", "mean"), monetary=("monetary", "mean"))
    names = name_segments(profile)
    profile["segment_name"] = [names[s][0] for s in profile.index]
    rep("\n## Task 3 - segment profiles (cluster numbers are labels only, with no order)")
    rep(profile.round(2).to_string())
    for s in profile.index:
        rep(f"segment {s} = {names[s][0]}: {names[s][1]}")

    premium_id = profile["monetary"].idxmax()
    rfm["segment_name"] = rfm["segment_id"].map(lambda s: names[s][0])
    rfm["is_premium"] = rfm["segment_id"] == premium_id
    rep(f"Premium = segment {premium_id} ({names[premium_id][0]}), highest mean monetary "
        f"INR {profile.loc[premium_id, 'monetary']:,.2f}; {int(rfm.is_premium.sum())} customers flagged")

    pca = PCA(n_components=2).fit(X)
    r = pca.explained_variance_ratio_
    rep(f"PCA(n_components=2) on scaled RFM: explained_variance_ratio_ = {r[0]:.3f}, {r[1]:.3f} (total {r.sum():.3f})")
    rep("PC loadings (recency, frequency, monetary): " +
        "; ".join(f"PC{i + 1} " + ", ".join(f"{v:+.2f}" for v in comp) for i, comp in enumerate(pca.components_)))

    cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    rfm[cols].to_csv(os.path.join(OUT, "customer_segments.csv"), index=False)
    rep(f"wrote outputs/customer_segments.csv ({len(rfm):,} rows)")
    rep.save()


if __name__ == "__main__":
    main()
