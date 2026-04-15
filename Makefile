venv:
	@python3 -m venv .venv

activate:
	@source .venv/bin/activate

dependencies:
	@python -m pip install -r requirements.txt

test:
	@python -m pytest -v

run:
	@python -m main

.PHONY: venv activate dependencies test run
