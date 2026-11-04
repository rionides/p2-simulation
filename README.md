# Wainwright sharp threshold simulation

#### How to run

Requires Python >= 3.12, with extra requirements in `requirements.txt`. On Mac all of these can be downloaded by

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make all
```

`make all` runs the entire simulation study detailed in the ADEMP (86,400 trials across 432 conditions), summarizes the saved trials, and produces the figures in results/figures. Note that running this command overwrites the selected studies' existing raw outputs. To rebuild figures from saved data
without running simulations, use `make figures`.

The stages can also run separately:

```sh
make simulate                  # run all studies and save trials
make summarize                 # make summary csv
make figures                   # generate figures
make test                      # unit tests
```

For a small, 90-trial check that this repo is running, run

```sh
make simulate CONFIG=quick
make summarize CONFIG=quick
```

`simulate` and `summarize` accept `CONFIG=all` (the default), `full`, `weak_signals`, or `quick`. One can edit the ordinary dictionaries in
[src/config.py](src/config.py) to change settings.

The simulation, summary, and figure targets accept `RESULTS=...` to change the output root. All targets accept `PYTHON=...` to choose an interpreter. For example, `make all RESULTS=another_run` keeps a separate complete run. Interrupted simulations must be restarted, as summaries
and figures refuse incomplete runs.

#### Directory layout

```text
.
├── ADEMP.md
├── ANALYSIS.md
├── Makefile
├── README.md
├── requirements.txt
├── results
│   ├── figures/
│   ├── raw/
│   └── tables/
├── src
│   ├── config.py
│   ├── dgps.py
│   ├── methods.py
│   ├── metrics.py
│   ├── simulation.py
│   └── visualize.py
└── tests
    ├── __init__.py
    ├── test_metrics.py
    ├── test_outputs.py
    └── test_simulation.py

```
