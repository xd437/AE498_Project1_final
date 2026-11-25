# Sparse identification of the Rayleigh–Plesset equation from spark-generated cavitation bubbles

AE498 Project 1, Chenghao Xu, University of Illinois Urbana-Champaign.

This folder contains the code and processed data needed to reproduce all results and figures of the report. The method is sparse regression (SINDy / PDE-FIND approach) applied to the measured first-cycle radius R(t) of 19 spark-generated cavitation bubbles.

## How to run

Requires Python 3.9 or newer with NumPy, SciPy, Matplotlib and openpyxl. If they are not
installed yet:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python rp_sindy_radius_final.py        # prints all numbers in Section IV of the report (a few seconds)
python make_figures_final.py           # writes fig_radius.pdf (Fig. 1) and fig_results.pdf (Fig. 2)
```
