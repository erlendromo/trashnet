.venv:
	@python3 -m venv .venv

.venv/installed: requirements.txt | .venv
	@.venv/bin/python -m pip install --upgrade pip
	@.venv/bin/python -m pip install -r requirements.txt
	@touch .venv/installed

test: .venv/installed
	@.venv/bin/python -m pytest -v

run: .venv/installed
	@.venv/bin/python -m main

.PHONY: test run
