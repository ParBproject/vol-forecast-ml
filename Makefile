PY ?= .venv/bin/python
.PHONY: all verify analyze readme lint test fetch
all: verify analyze readme web-data lint test
fetch:        ## ONLINE: re-download raw data (changes hashes; not part of `all`)
	$(PY) scripts/fetch_data.py && $(PY) scripts/manifest.py
verify:
	$(PY) scripts/manifest.py --check
analyze:      ## full walk-forward study, ~4 min on CPU
	PYTHONPATH=src $(PY) -m volml.analysis
readme:
	PYTHONPATH=src $(PY) scripts/build_readme.py
lint:
	$(PY) -m ruff check .
test:
	$(PY) -m pytest -q

## Web app (needs Node 20+).
.PHONY: web-data web-test web deploy
web-data:
	$(PY) scripts/build_web_data.py
web-test:
	cd web && npm ci --silent && npm test
web: web-data web-test
	cd web && npm run build
deploy: web    ## publish web/dist to the gh-pages branch
	bash scripts/deploy_pages.sh

## ONLINE launch check (not part of `all`): README screenshots (Playwright).
.PHONY: shots
shots:
	$(PY) scripts/shoot_readme.py
