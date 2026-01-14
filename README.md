# Robotics ArXiv Daily (Autonomous Vehicles • Drones • 3D Gaussian Splatting • More)

A lightweight **daily ArXiv digest** for robotics-related papers, including:
- **Autonomous driving** (perception / prediction / planning / BEV / mapping)
- **Drones / aerial robotics** (navigation, control, SLAM, perception)
- **General robotics** (manipulation, navigation, learning, safety)
- **3D Gaussian Splatting / Neural Rendering for robotics**
- **SLAM / Localization / Mapping**
- **Multi-robot / Swarms**
- **Robustness / Uncertainty / Safety**

This repo is designed to run **automatically** via GitHub Actions and update the digest every day.

---

## Quick Start (for non-experts)

### 1) Create your GitHub repository
- Create a new repo on GitHub (e.g. `robotics-arxiv-daily`).
- Upload the contents of this project (or push via git).

### 2) Enable GitHub Actions
- Go to **Actions** tab → enable workflows if GitHub asks.

### 3) Wait for the scheduled run (or run manually)
- The workflow runs daily.
- You can also run it right now:
  - Actions → **Daily ArXiv Digest** → **Run workflow**

The digest will appear in:
- `README.md` (top “Today” section)
- `digests/YYYY-MM-DD.md` (daily archive files)

---

## Configuration (what gets collected)

Edit **`config.yml`**:
- `categories`: arXiv categories to search (e.g. `cs.RO`, `cs.CV`, `eess.IV`)
- `topics`: grouped keyword filters (autonomous driving, drones, 3DGS, SLAM, …)
- `max_results_per_topic`: cap per topic per day
- `days_back`: how many days to consider as “new”

> Tip: If you want fewer papers, reduce `days_back` to 1 and lower `max_results_per_topic`.

---

## Local Run (optional)

### Install
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Run
```bash
python scripts/fetch_arxiv_daily.py
```

---

## Output Format
Each topic produces a section like:
- Title
- Authors
- Published date
- arXiv link + PDF link
- Primary category + matched keywords

---

## License
Apache-2.0 (recommended for broad re-use)

---

## Credits
Built as a simple, reproducible “ArXiv daily” style digest using the arXiv Atom API.
