import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA, OUT = Path("data"), Path("results")
OUT.mkdir(exist_ok=True)
SEED = 42 
random.seed(SEED); np.random.seed(SEED)

BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#9a9a94", "#2b2b28"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False,
})
summary = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {name}")


print("[1] loading")
nodes = pd.read_csv(DATA / "nodes.csv")
edges = pd.read_csv(DATA / "edges.csv")
removed = nodes.loc[nodes["on_pypi"] != True, "id"].tolist()      
nodes = nodes[nodes["on_pypi"] == True].copy()     
edges = edges[edges["source"].isin(nodes["id"]) & edges["target"].isin(nodes["id"])]

G = nx.DiGraph()
G.add_nodes_from(nodes["id"])
G.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
U = G.to_undirected()
summary["removed_packages"] = removed

print("[2] basic structure")
n, m = G.number_of_nodes(), G.number_of_edges()
wcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
scc_big = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
giant = G.subgraph(wcc[0]).copy()
summary["structure"] = {
    "nodes": n, "edges": m,
    "density": nx.density(G),
    "mean_out_degree": m / n,
    "isolated_nodes": sum(1 for v in G if G.degree(v) == 0),
    "weak_components": len(wcc),
    "giant_component_nodes": len(wcc[0]),
    "giant_component_share": len(wcc[0]) / n,
    "second_component_size": len(wcc[1]) if len(wcc) > 1 else 0,
    "is_DAG": nx.is_directed_acyclic_graph(G),
    "cycles_strong_components_gt1": len(scc_big),
    "nodes_in_cycles": sum(len(c) for c in scc_big),
    "cycle_groups": [sorted(c) for c in scc_big],
    "reciprocity": nx.reciprocity(G),
    "longest_dependency_chain": nx.dag_longest_path_length(nx.condensation(G)),
}
sizes = pd.Series([len(c) for c in wcc]).value_counts().sort_index()
sizes.rename_axis("component_size").rename("count").to_csv(OUT / "component_sizes.csv")

print("[3] degree distributions")
indeg = pd.Series(dict(G.in_degree()))
outdeg = pd.Series(dict(G.out_degree()))


def describe(s):
    return {"mean": s.mean(), "median": float(s.median()), "max": int(s.max()),
            "share_zero": float((s == 0).mean()), "top1pct_share_of_links": float(s.nlargest(max(1, len(s)//100)).sum() / s.sum())}


def powerlaw_alpha(s, kmin):
    """Discrete power-law exponent, maximum-likelihood approximation (Clauset et al. 2009, eq. 3.7)."""
    k = s[s >= kmin].astype(float)
    return 1 + len(k) / np.log(k / (kmin - 0.5)).sum(), len(k)


def best_kmin(s, candidates=range(1, 30)):
    """Choose kmin by minimising the KS distance between data and fitted power law."""
    best = None
    for kmin in candidates:
        k = np.sort(s[s >= kmin].values)
        if len(k) < 50:
            break
        a, _ = powerlaw_alpha(s, kmin)
        emp = 1 - np.arange(len(k)) / len(k)                 
        fit = (k / kmin) ** (1 - a)                            
        ks = np.abs(emp - fit).max()
        if best is None or ks < best[2]:
            best = (kmin, a, ks)
    return best


summary["in_degree"] = describe(indeg)
summary["out_degree"] = describe(outdeg)
for name, s in (("in_degree", indeg), ("out_degree", outdeg)):
    kmin, alpha, ks = best_kmin(s)
    summary[name].update({"powerlaw_kmin": kmin, "powerlaw_alpha": alpha, "powerlaw_KS": ks,
                          "n_tail": int((s >= kmin).sum())})

fig, ax = plt.subplots(figsize=(5.5, 4))
for s, col, lab in ((indeg, BLUE, "In-degree (packages that need it)"),
                    (outdeg, ORANGE, "Out-degree (packages it needs)")):
    pos = s[s > 0]
    k = np.sort(pos.unique())
    ccdf = np.array([(pos >= x).mean() for x in k])         
    ax.loglog(k, ccdf, marker="o", ms=3, lw=0, color=col, label=lab, alpha=0.8)
ax.set_xlabel("Degree k")
ax.set_ylabel("Fraction of packages with degree ≥ k")
ax.set_title("Degree distributions (log-log CCDF)", loc="left")
ax.legend()
save(fig, "fig1_degree_ccdf.png")

# CENTRALITY
print("[4] centrality (betweenness takes a minute or two)")
pagerank = pd.Series(nx.pagerank(G, alpha=0.85))
betweenness = pd.Series(nx.betweenness_centrality(G, normalized=True))

harmonic = pd.Series(nx.harmonic_centrality(G)) / (n - 1)

try:
    ev = pd.Series(nx.eigenvector_centrality(G.reverse(), max_iter=1000))
    summary["eigenvector_nonzero_share"] = float((ev > 1e-6).mean())
    summary["eigenvector_note"] = ("only this share of packages gets a non-zero score, so eigenvector "
                                   "centrality is not usable here (L05) - PageRank used instead")
except nx.PowerIterationFailedConvergence:
    summary["eigenvector_note"] = "did not converge (network almost acyclic) - PageRank used instead"

meas = nodes.set_index("id")[["download_rank", "downloads_30d", "in_seed", "topic", "license",
                              "first_release", "n_releases", "summary"]].copy()
meas["in_degree"] = indeg
meas["out_degree"] = outdeg
meas["pagerank"] = pagerank
meas["betweenness"] = betweenness
meas["harmonic_in"] = harmonic
meas["wcc_giant"] = meas.index.isin(wcc[0])
meas["in_cycle"] = meas.index.isin(set().union(*scc_big) if scc_big else set())
for col in ("in_degree", "out_degree", "pagerank", "betweenness", "harmonic_in"):
    meas[f"{col}_rank"] = meas[col].rank(ascending=False, method="min").astype(int)
meas.sort_values("pagerank", ascending=False).to_csv(OUT / "node_measures.csv")

cols = ["download_rank", "in_degree", "out_degree", "pagerank", "betweenness", "topic"]
for c in ("in_degree", "pagerank", "betweenness", "out_degree"):
    meas.nlargest(15, c)[cols].to_csv(OUT / f"top_{c}.csv")

top = meas.nlargest(15, "pagerank")
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.barh(top.index[::-1], top["pagerank"][::-1], color=BLUE, height=0.6)
ax.set_xlabel("PageRank")
ax.set_title("Top 15 packages by PageRank", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "fig2_top_pagerank.png")

# how much do the measures agree? Spearman rank correlation
corr = meas[["in_degree", "pagerank", "betweenness", "harmonic_in", "out_degree"]].corr("spearman")
corr.to_csv(OUT / "centrality_correlation.csv")
summary["top10_overlap_indegree_pagerank"] = len(set(meas.nlargest(10, "in_degree").index) &
                                                 set(meas.nlargest(10, "pagerank").index))

# relation of the download and structure of the netowrk
print("[5] downloads vs structure")
seeded = meas[meas["downloads_30d"].notna()].copy()
summary["spearman_downloads_vs"] = {
    c: float(seeded["downloads_30d"].corr(seeded[c], method="spearman"))
    for c in ("in_degree", "pagerank", "betweenness", "out_degree")
}
# Hidden infrastructure: In the top 200 by PageRank, but with the worst download ranks
seeded[seeded["pagerank_rank"] <= 200].nlargest(15, "download_rank")[
    ["download_rank", "pagerank_rank", "in_degree", "summary"]].to_csv(
    OUT / "hidden_infrastructure.csv")
# reverse: hugely downloaded but almost nobody depends on them 
seeded[seeded["download_rank"] <= 200].nsmallest(15, "in_degree")[
    ["download_rank", "in_degree", "out_degree", "summary"]].to_csv(OUT / "popular_but_not_depended_on.csv")

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(seeded["downloads_30d"], seeded["in_degree"] + 1, s=8, color=BLUE, alpha=0.4, linewidths=0)
for i, name in enumerate(seeded.nlargest(3, "in_degree").index):   # label only the top 3
    ax.annotate(name, (seeded.loc[name, "downloads_30d"], seeded.loc[name, "in_degree"] + 1),
                fontsize=8, color=INK, xytext=(-8, 6 - 12 * i), textcoords="offset points", ha="right")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Downloads in 30 days")
ax.set_ylabel("In-degree + 1")
ax.set_ylim(top=3000)
ax.set_title(f"Downloads vs packages that depend on it "
             f"(Spearman ρ = {summary['spearman_downloads_vs']['in_degree']:.2f})", loc="left", fontsize=9.5)
save(fig, "fig3_downloads_vs_indegree.png")

#SMALL-WORLD TEST
print("[6] small-world test on the giant component (undirected)")
Ug = U.subgraph(wcc[0]).copy()


def avg_path(g, samples=600):
    """Average shortest path length; exact for small graphs, sampled BFS otherwise."""
    nodes_ = list(g)
    src = nodes_ if len(nodes_) <= samples else random.sample(nodes_, samples)
    tot = cnt = 0
    for s in src:
        d = nx.single_source_shortest_path_length(g, s)
        tot += sum(d.values()); cnt += len(d) - 1
    return tot / cnt


C, L = nx.average_clustering(Ug), avg_path(Ug)
Cr, Lr = [], []
for i in range(5):
    R = nx.gnm_random_graph(Ug.number_of_nodes(), Ug.number_of_edges(), seed=SEED + i)
    R = R.subgraph(max(nx.connected_components(R), key=len)).copy()
    Cr.append(nx.average_clustering(R)); Lr.append(avg_path(R))
summary["small_world"] = {
    "C": C, "L": L, "C_random": float(np.mean(Cr)), "L_random": float(np.mean(Lr)),
    "sigma": (C / np.mean(Cr)) / (L / np.mean(Lr)),
    "log_n": float(np.log(Ug.number_of_nodes())),
    "diameter_estimate": nx.approximation.diameter(Ug),
    "note": "computed on the undirected giant component; sigma > 1 suggests small-world",
}

# Robustness
print("[7] robustness: random failures vs removing hubs")
steps = np.linspace(0, 0.10, 21)


def giant_share(g, order):
    out = []
    for f in steps:
        h = g.copy()
        h.remove_nodes_from(order[: int(f * n)])
        out.append(len(max(nx.connected_components(h), key=len)) / n if h.number_of_nodes() else 0)
    return out


hub_order = list(indeg.sort_values(ascending=False).index)
rand_runs = [giant_share(U, random.sample(list(U), n)) for _ in range(5)]
rand_curve = np.mean(rand_runs, axis=0)
hub_curve = giant_share(U, hub_order)
pd.DataFrame({"fraction_removed": steps, "random": rand_curve, "hubs_first": hub_curve}).to_csv(
    OUT / "robustness.csv", index=False)
summary["robustness_giant_share_after_removing_1pct"] = {
    "random": float(rand_curve[2]), "hubs_first": float(hub_curve[2])}

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(steps * 100, rand_curve, color=BLUE, lw=2, label="Random packages removed")
ax.plot(steps * 100, hub_curve, color=ORANGE, lw=2, label="Most-depended-on removed first")
ax.set_xlabel("% of packages removed")
ax.set_ylabel("Share of packages in largest component")
ax.set_ylim(0, 1)
ax.set_title("Robustness of the network", loc="left")
ax.legend()
save(fig, "fig4_robustness.png")

print("[8] assortativity and k-cores")

# degree assortativity: do hubs connect to hubs, or small packages to hubs?
summary["degree_assortativity"] = {
    "undirected": nx.degree_assortativity_coefficient(U),
    "source_out_vs_target_in": nx.degree_assortativity_coefficient(G, x="out", y="in"),
    "source_in_vs_target_in": nx.degree_assortativity_coefficient(G, x="in", y="in"),
    "note": "r<0 = small packages tend to depend on big hubs (disassortative)",
}

# assortativity: do packages depend on packages from their own topic?
topic = nodes.set_index("id")["topic"].dropna()
labelled = [v for v in G if v in topic.index]
Gt = G.subgraph(labelled).copy()
nx.set_node_attributes(Gt, topic.to_dict(), "topic")
Ut = Gt.to_undirected()
groups = {}
for v, t in topic.items():
    if v in Ut:
        groups.setdefault(t, set()).add(v)
same = sum(1 for a, b in Gt.edges() if topic[a] == topic[b])
summary["topic_assortativity"] = {
    "labelled_nodes": Gt.number_of_nodes(), "labelled_share": Gt.number_of_nodes() / n,
    "edges_between_labelled": Gt.number_of_edges(),
    "share_edges_same_topic": same / max(1, Gt.number_of_edges()),
    "assortativity_r": nx.attribute_assortativity_coefficient(Gt, "topic"),
    "modularity_Q_topic_partition": nx.community.modularity(Ut, groups.values()),
    "n_topics": len(groups),
    "note": "r and Q > 0 = more same-topic links than expected at random",
}

mix = pd.crosstab(pd.Series([topic[a] for a, b in Gt.edges()], name="from_topic"),
                  pd.Series([topic[b] for a, b in Gt.edges()], name="to_topic"))
mix.to_csv(OUT / "topic_mixing_counts.csv")
main_topics = topic[topic.index.isin(labelled)].value_counts().head(8).index
sub = mix.reindex(index=main_topics, columns=main_topics, fill_value=0)
share = sub.div(sub.sum(axis=1).replace(0, 1), axis=0)      

fig, ax = plt.subplots(figsize=(6.2, 5))
im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(main_topics)), main_topics, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(main_topics)), main_topics, fontsize=8)
for i in range(len(main_topics)):
    for j in range(len(main_topics)):
        v = share.values[i, j]
        if v >= 0.05:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else INK)
ax.set_xlabel("Topic of the package that is needed")
ax.set_ylabel("Topic of the package that depends")
ax.set_title("Who depends on whom, by topic (row shares)", loc="left", fontsize=10)
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
save(fig, "fig5_topic_mixing.png")

# k cores: peel away weakly linked packages until only the dense centre is left
core = pd.Series(nx.core_number(U))
kmax = int(core.max())
inner = core[core == kmax].index
top20_pr = set(meas.nlargest(20, "pagerank").index)
summary["k_core"] = {
    "max_k": kmax,
    "innermost_core_size": len(inner),
    "innermost_core_members": sorted(inner),
    "top20_pagerank_in_innermost_core": len(top20_pr & set(inner)),
    "nodes_per_k": core.value_counts().sort_index().to_dict(),
}
meas["core_number"] = core
meas.sort_values("pagerank", ascending=False).to_csv(OUT / "node_measures.csv")  # rewrite with new column

counts = core.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
ax.set_xlabel("Core number k (package survives peeling up to k)")
ax.set_ylabel("Number of packages")
ax.set_title(f"k-core structure (innermost core: k = {kmax}, {len(inner)} packages)", loc="left", fontsize=10)
ax.grid(axis="x", visible=False)
save(fig, "fig6_kcore.png")


(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print("\nDone. Headline numbers:")
print(json.dumps({k: v for k, v in summary.items() if k != "structure"}, indent=2, default=float)[:3000])
print(json.dumps({k: v for k, v in summary["structure"].items() if k != "cycle_groups"}, indent=2, default=float))
