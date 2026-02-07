# Bayesian Sequential Decision-Making Thesis
# Makefile for reproducibility

.PHONY: install test run thesis clean all smoke

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v --tb=short

# Run smoke test (quick validation)
smoke:
	python examples/run_phase1.py --smoke

# Run full baseline experiment
run:
	python examples/run_phase1.py

# Compile thesis to PDF
thesis:
	cd thesis && pdflatex -interaction=nonstopmode main.tex && \
	bibtex main && \
	pdflatex -interaction=nonstopmode main.tex && \
	pdflatex -interaction=nonstopmode main.tex

# Clean build artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	cd thesis && rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot

# Full pipeline: install, test, run, compile thesis
all: install test run thesis

# Help
help:
	@echo "Available targets:"
	@echo "  install  - Install Python dependencies"
	@echo "  test     - Run pytest test suite"
	@echo "  smoke    - Quick validation (10 runs)"
	@echo "  run      - Full baseline experiment (100 runs)"
	@echo "  thesis   - Compile LaTeX thesis to PDF"
	@echo "  clean    - Remove build artifacts"
	@echo "  all      - Full pipeline"
