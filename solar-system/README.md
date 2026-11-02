# Solar System — IDAC-BR V

Hands-on for the **3rd LSST@LATAM Meeting** (ICTP-SAIFR): occultations via LIneA Portal / `lineaSSP`, then TAP queries on public `mpc_sbn` and restricted `lsst_dp1` / `lsst_dp2`.

## Notebooks

| Notebook | Focus |
| --- | --- |
| `tutorial_occultations.ipynb` | HTTP / `lineaSSP` (occultation predictions) |
| `tutorial_tap.ipynb` | TAP SQL: `mpc_sbn` / `X05`, light curves, DP1/DP2 |

Run occultations first, then TAP. If a live cell fails, use the fallback cell under it (`data/` + `utils.py`).

## Setup

```bash
cd solar-system
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook tutorial_occultations.ipynb
```

On-site: [JupyterHub](https://jupyter.linea.org.br) with the **SolarSystem** image.  
More detail: [`notes/README.md`](notes/README.md).
