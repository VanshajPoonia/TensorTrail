PYTHON ?= $(shell if [ -x .venv/bin/python ]; then printf ".venv/bin/python"; else command -v python3 2>/dev/null || command -v python; fi)
PIP ?= $(PYTHON) -m pip

.PHONY: install test xor examples

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest

xor:
	$(PYTHON) examples/train_xor.py

examples:
	$(PYTHON) examples/train_xor.py
	$(PYTHON) examples/train_mnist_like.py
	$(PYTHON) examples/train_mlp_classifier.py
	$(PYTHON) examples/train_regularized_mlp.py
	$(PYTHON) examples/train_tiny_cnn.py
	$(PYTHON) examples/visualize_autograd_graph.py
