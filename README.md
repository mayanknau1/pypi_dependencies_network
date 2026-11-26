# Who holds up the Python world?
### A network analysis of package dependencies on PyPI

Exam project for **Network Analysis**, University of Bologna, 2026/27.

## What this is

Every Python package depends on other packages. If one small package breaks or disappears, everything built on top of it can break too. 
We turned the most downloaded packages on pypi into a network and asked one main question:

> **Which packages hold up the Python ecosystem, and is the most critical package the same as the most popular one?**

- **Nodes:** packages
- **Edges:** `A→B` means "package A depends on package B"
- **In-degree of B:** how many packages need B

## Repository layout

```
.
├── collect_pypi.py    
├── analyse_pypi.py     
├── reference papers
├── data
│   ├── raw
│   │   ├── top_pypi_packages.json  
│   │   └── pypi_json/             
│   ├── nodes.csv        
│   ├── edges.csv         
│   ├── edges_all.csv     
│   ├── pypi_deps.graphml 
│   └── collection_log.json
├── results
│   ├── summary.json  
│   ├── node_measures.csv 
│   ├── top_.csv       
│   └── fig.png           
└── report
    └── report.pdf         
```

## How to reproduce

```bash
pip install -r requirements.txt

# 1. Collect the data (about 15–30 minutes; try top 200 first
python collect_pypi.py top 5000, expand with some other important dependencies 

# 2. Run the analysis (about 2–5 minutes)
python analyse_pypi.py
```

## Data sources

- **PyPI JSON API:** `https://pypi.org/pypi/<name>/json`, field `requires_dist`
- **Top PyPI packages:** list of the most downloaded packages by Hugo van Kemenade ([hugovk/top-pypi-packages](https://github.com/hugovk/top-pypi-packages))

Optional dependencies (extras such as `pandas[excel]`) are kept in `edges_all.csv` but left out of the main network in `edges.csv`.


## Authors

- Mayank Nautiyal
- Gaurav Gautam

