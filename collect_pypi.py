"""
collect_pypi.py - Build the PyPI dependency network for the Network Analysis project.

What it does (one step per function):
  1. get_seed_list()     -> the N most-downloaded packages (hugovk/top-pypi-packages)
  2. fetch_metadata()    -> PyPI JSON for every package, cached in data/raw/pypi_json/
  3. parse_dependencies()-> turn each "requires_dist" line into an edge
  4. build_tables()      -> nodes.csv, edges_all.csv, edges.csv, pypi_deps.graphml

Edge direction:  A -> B  means "package A depends on package B".
So in-degree of B = number of packages that need B (its "popularity as a dependency").

Usage:
  pip install requests packaging pandas networkx
  python collect_pypi.py --top 5000                 # seeds only (induced subgraph)
  python collect_pypi.py --top 5000 --expand        # also pull in deps outside the top list
  python collect_pypi.py --seed-file my_list.txt    # use your own list (one name per line)
"""

import argparse
import json
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pandas as pd
import requests
from packaging.requirements import InvalidRequirement, Requirement

DATA = Path("data")
RAW = DATA / "raw"
CACHE = RAW / "pypi_json"
TOP_URLS = [  # the file was renamed over the years; try both
    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json",
    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json",
]
HEADERS = {"User-Agent": "unibo-dhdk-network-analysis-student-project"}


def normalize(name: str) -> str:
    """PEP 503 name normalisation: 'Scikit_Learn' and 'scikit-learn' are the same node."""
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------------------------------------------------------------- STEP 1
def get_seed_list(top_n: int, seed_file: str | None) -> pd.DataFrame:
    if seed_file:
        names = [l.strip() for l in Path(seed_file).read_text().splitlines() if l.strip()]
        return pd.DataFrame({"name": [normalize(n) for n in names],
                             "download_rank": range(1, len(names) + 1),
                             "downloads_30d": None})
    out = RAW / "top_pypi_packages.json"
    if not out.exists():
        for url in TOP_URLS:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.ok:
                out.write_text(r.text)
                break
        else:
            raise SystemExit("Could not download the top-packages list. Download it by hand into data/raw/.")
    payload = json.loads(out.read_text())
    rows = payload["rows"][:top_n]
    print(f"[1] seed list: {len(rows)} packages (snapshot: {payload.get('last_update')})")
    return pd.DataFrame({"name": [normalize(r["project"]) for r in rows],
                         "download_rank": range(1, len(rows) + 1),
                         "downloads_30d": [r["download_count"] for r in rows]})


# ---------------------------------------------------------------- STEP 2
def fetch_one(name: str) -> dict | None:
    """Get https://pypi.org/pypi/<name>/json, using the local cache if present."""
    path = CACHE / f"{name}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data or None
    for attempt in range(3):
        try:
            r = requests.get(f"https://pypi.org/pypi/{name}/json", headers=HEADERS, timeout=30)
            if r.status_code == 404:          # deleted / never existed
                path.write_text("{}")
                return None
            r.raise_for_status()
            data = r.json()
            # keep only what we need (the full 'releases' block is huge)
            slim = {
                "info": data["info"],
                "release_dates": {
                    v: min(f["upload_time_iso_8601"] for f in files)
                    for v, files in data.get("releases", {}).items() if files
                },
            }
            path.write_text(json.dumps(slim))
            return slim
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def fetch_metadata(seeds: list[str], expand: bool, max_nodes: int, workers: int) -> dict:
    """Download metadata. With --expand, follow dependencies (BFS) until closure or max_nodes."""
    meta, seen, queue = {}, set(seeds), deque(seeds)
    while queue:
        batch = [queue.popleft() for _ in range(min(len(queue), 200))]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_one, n): n for n in batch}
            for fut in as_completed(futures):
                name = futures[fut]
                meta[name] = fut.result()
                if expand and meta[name]:
                    for dep in runtime_dep_names(meta[name]):
                        if dep not in seen and len(seen) < max_nodes:
                            seen.add(dep)
                            queue.append(dep)
        print(f"[2] fetched {len(meta)} / {len(seen)} known packages")
    return meta


# ---------------------------------------------------------------- STEP 3
def parse_requirement(line: str):
    try:
        req = Requirement(line)
    except InvalidRequirement:
        return None
    marker = str(req.marker) if req.marker else ""
    return {
        "target": normalize(req.name),
        "specifier": str(req.specifier),          # e.g. ">=2.0,<3"
        "marker": marker,                         # e.g. python_version < "3.8"
        "optional": "extra" in marker,            # only installed with pip install pkg[extra]
        "extra": re.findall(r'extra\s*==\s*["\']([^"\']+)', marker)[0] if "extra" in marker else "",
        "conditional": bool(marker) and "extra" not in marker,   # platform / python-version deps
    }


def runtime_dep_names(m: dict) -> set[str]:
    deps = set()
    for line in m["info"].get("requires_dist") or []:
        p = parse_requirement(line)
        if p and not p["optional"]:
            deps.add(p["target"])
    return deps


def parse_dependencies(meta: dict) -> pd.DataFrame:
    rows = []
    for src, m in meta.items():
        if not m:
            continue
        for line in m["info"].get("requires_dist") or []:
            p = parse_requirement(line)
            if p and p["target"] != src:          # drop self-loops (pkg[extra] -> pkg)
                rows.append({"source": src, **p})
    edges = pd.DataFrame(rows).drop_duplicates(subset=["source", "target", "optional", "extra"])
    print(f"[3] parsed {len(edges)} dependency lines")
    return edges


# ---------------------------------------------------------------- STEP 4
def top_classifier(classifiers, prefix):
    vals = [c.split(" :: ")[1] for c in classifiers if c.startswith(prefix) and c.count(" :: ") >= 1]
    return vals[0] if vals else ""


def build_tables(seed_df: pd.DataFrame, meta: dict, edges: pd.DataFrame, expand: bool):
    seed_set = set(seed_df["name"])
    node_rows = []
    for name, m in meta.items():
        if not m:
            node_rows.append({"id": name, "on_pypi": False})
            continue
        info, dates = m["info"], m["release_dates"]
        cls = info.get("classifiers") or []
        node_rows.append({
            "id": name,
            "display_name": info["name"],
            "on_pypi": True,
            "in_seed": name in seed_set,
            "latest_version": info["version"],
            "first_release": min(dates.values()) if dates else "",
            "latest_release": dates.get(info["version"], ""),
            "n_releases": len(dates),
            "requires_python": info.get("requires_python") or "",
            "license": (info.get("license_expression") or info.get("license") or "")[:60],
            "topic": top_classifier(cls, "Topic ::"),
            "dev_status": top_classifier(cls, "Development Status ::"),
            "summary": (info.get("summary") or "")[:200],
        })
    nodes = pd.DataFrame(node_rows).merge(seed_df, left_on="id", right_on="name", how="left").drop(columns="name")

    # keep edges only between packages we actually have as nodes
    known = set(nodes.loc[nodes["on_pypi"] == True, "id"])  # noqa: E712
    edges["target_in_network"] = edges["target"].isin(known)
    edges.to_csv(DATA / "edges_all.csv", index=False)               # everything, for transparency

    main = edges[(~edges["optional"]) & edges["target_in_network"]]
    main = main.drop_duplicates(subset=["source", "target"])
    main[["source", "target", "specifier", "conditional"]].to_csv(DATA / "edges.csv", index=False)

    # graph + simple degree columns so nodes.csv is useful on its own
    G = nx.DiGraph()
    G.add_nodes_from(known)
    G.add_edges_from(main[["source", "target"]].itertuples(index=False, name=None))
    nodes["in_degree"] = nodes["id"].map(dict(G.in_degree())).fillna(0).astype(int)
    nodes["out_degree"] = nodes["id"].map(dict(G.out_degree())).fillna(0).astype(int)
    nodes.to_csv(DATA / "nodes.csv", index=False)

    for _, r in nodes[nodes["on_pypi"] == True].iterrows():  # noqa: E712
        G.nodes[r["id"]].update({k: ("" if pd.isna(v) else v) for k, v in r.items()
                                 if k in ("topic", "license", "first_release", "download_rank", "in_seed")})
    nx.write_graphml(G, DATA / "pypi_deps.graphml")              # opens in Gephi

    log = {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed_size": len(seed_set), "expanded": expand,
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "missing_from_pypi": int((nodes["on_pypi"] == False).sum()),  # noqa: E712
        "optional_edges_dropped": int(edges["optional"].sum()),
        "edges_to_outside_dropped": int((~edges["target_in_network"] & ~edges["optional"]).sum()),
    }
    (DATA / "collection_log.json").write_text(json.dumps(log, indent=2))
    print("[4] done:", json.dumps(log, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5000)
    ap.add_argument("--seed-file")
    ap.add_argument("--expand", action="store_true", help="follow deps outside the seed list")
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    seeds = get_seed_list(args.top, args.seed_file)
    meta = fetch_metadata(list(seeds["name"]), args.expand, args.max_nodes, args.workers)
    edges = parse_dependencies(meta)
    build_tables(seeds, meta, edges, args.expand)


if __name__ == "__main__":
    main()
