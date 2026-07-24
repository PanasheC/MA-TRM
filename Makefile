.PHONY: test smoke params benchmark reproduce lint clean

test:
	pytest -q

smoke:
	python scripts/smoke_test.py --topology sequential
	python scripts/smoke_test.py --topology mixture
	python scripts/smoke_test.py --topology deliberation
	python scripts/smoke_test.py --topology distillation

params:
	python scripts/count_parameters.py --output benchmarks/parameter_budget.json

benchmark:
	python scripts/benchmark_models.py

reproduce:
	./scripts/run_reproducibility_suite.sh

lint:
	ruff check models train scripts tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
