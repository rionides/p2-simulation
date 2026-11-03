# Wainwright sharp threshold simulation

A simple Python reproduction of both experiments in Section 4 of
[Wainwright's paper](Wainwright_SharpThreshold.pdf), plus a weak-signal
extension. See [ADEMP.md](ADEMP.md) for the full scientific design.
See [ANALYSIS.md](ANALYSIS.md) for the findings, reproduction limits, and critique of the design.

**All outputs live in `results/`. Open [results/figures](results/figures)
for the three final figures.** Raw trials, derived tables, and figures have
separate folders.

## Run

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

`simulate` and `summarize` accept `CONFIG=all` (the default), `full`,
`weak_signals`, or `quick`. One can edit the ordinary dictionaries in
[src/config.py](src/config.py) to change settings.

All stages accept `RESULTS=...` to change the output root and `PYTHON=...` to
choose an interpreter. For example, `make all RESULTS=another_run` keeps a
separate complete run. Interrupted simulations must be restarted, as summaries
and figures refuse incomplete runs.

## Directory layout

```text
src/
  config.py          Parameter dictionaries
  dgps.py            Sparsity, sample size, and data generation
  methods.py         Lasso penalty and fitting
  metrics.py         Recovery, summaries, intervals, and failure modes
  simulation.py      Generate, fit, score, save, and summarize trials
  visualize.py       Load saved data and draw the three final figures
tests/               Unit tests and saved-output checks
results/
  raw/               Saved trials, settings, metadata, and logs by preset
  tables/            Study summaries and figure data
  figures/           Three final figures, each as PNG and PDF
```

The command-line entry points are `python -m src.simulation simulate`,
`python -m src.simulation summarize`, and `python -m src.visualize`.
The Makefile supplies their configuration and results-directory arguments.

## Study settings

- `full`: p = 128, 256, 512; three sparsity regimes; 43,200 trials.
- `weak_signals`: the same settings, with five active magnitudes changed from
  1 to 0.25; 43,200 trials. Paired trials retain the same predictors, support,
  signs, and noise draws. A fixed minimum magnitude of 0.25 satisfies the
  paper's asymptotic signal condition.

All studies use Gaussian noise with standard deviation 0.5 and independent
Gaussian predictor rows. Their covariance is either identity or
`Sigma[i,j] = 0.1**abs(i-j)`. Both designs come from the paper: identity in
Figure 1 and Toeplitz in Figure 2, p. 21, equation (39).
Each trial draws a uniformly random support and independent positive or
negative signs. Active magnitudes are 1 except for the five weak effects.

The sparsity rules are `k = floor(0.4*p)`,
`floor(0.4*p/log(0.4*p))`, and `floor(0.4*p**0.75)`. Each condition has 200
trials, with `theta = 0.1, 0.3, ..., 2.3`. Sample size follows Section 4's text:

```text
n = floor(2 * theta * k * log(p-k)) + k + 1
rho = sqrt(log(p-k) * log(k) / n)
Lasso objective = ||y - X @ beta||² / (2*n) + rho * sum(abs(beta))
```

All logs are natural. There is no fitted intercept, standardization, or
cross-validation. Success requires every estimated sign, including zeros, to
match the truth. Magnitudes at most `1e-7` count as zero; solver tolerance is
`1e-7`, with at most 10,000 iterations. Convergence warnings stop a run.

The base seed is **607**, with a new generator using `607 + zero_based_row_number`
for each trial. Trials run in configuration order: design, regime, dimension,
theta, then repetition. Changing that order changes seed assignments. Rounding,
the theta grid, solver details, and seeds are documented reproduction choices;
the new curves are not expected to match the paper point for point.
