PYTHON ?= python3
CONFIG ?= all
RESULTS ?= results
export MPLCONFIGDIR ?= $(CURDIR)/.mplconfig

.PHONY: all simulate summarize test figures

all:
	$(MAKE) simulate CONFIG=all
	$(MAKE) summarize CONFIG=all
	$(MAKE) figures

simulate:
	$(PYTHON) -m src.simulation simulate --config "$(CONFIG)" --results "$(RESULTS)"

summarize:
	$(PYTHON) -m src.simulation summarize --config "$(CONFIG)" --results "$(RESULTS)"

test:
	$(PYTHON) -m unittest discover -s tests -v

figures:
	$(PYTHON) -m src.visualize --results "$(RESULTS)"
