# Define variables
VENV = .venv
PYTHON = $(VENV)/bin/python
UV = uv

# Default target
.PHONY: all
all: install

# Create virtual environment
.PHONY: init
venv:
	uv init
	# python3 -m venv $(VENV)

# Install dependencies
.PHONY: install
install: venv
	$(UV) venv $(VENV)
	$(UV) pip install -r requirements.txt

# Run tests
.PHONY: test
test:
	$(PYTHON) -m unittest discover tests

# Format code using ruff
.PHONY: format
format:
	$(PYTHON) -m ruff format .

# Lint code using flake8
.PHONY: lint
lint:
	$(PYTHON) -m flake8 .

# Clean up generated files
.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
