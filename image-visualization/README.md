# Image Visualization — IDAC-BR V

Hands-on for the **3rd LSST@LATAM Meeting** (ICTP-SAIFR): asynchronous multi-band FITS cutouts from the LIneA Cutout Service (DES DR2), with a short photometry check on the downloaded files.

## Notebooks

| Notebook | Focus |
| --- | --- |
| `ic1816_cutout.ipynb` | Async LIneA cutouts: IC 1816, DES DR2 (`g, r, i, z, Y`) |

Band pairing and FITS cache live in `utils.py`. Cached files go under `cutouts/` — delete that folder to force a fresh download.

## Setup

```bash
cd image-visualization
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ic1816_cutout.ipynb
```

On-site: [JupyterHub](https://jupyter.linea.org.br).
