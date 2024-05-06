install-dependencies:
	pip install -r requirements.txt

prepare-code:
	black --line-length=119 d2dmoe
	flake8 d2dmoe --max-line-length=119
	black --line-length=119 tests
	flake8 tests --max-line-length=119

test:
	export PYTHONPATH=`pwd` && pytest

get-coverage:
	export PYTHONPATH=`pwd` && pytest -vv --cov=d2dmoe --cov-report=term-missing
	rm .coverage*

build-release:
	rm -rf dist
	rm -rf build
	python -m pip install --upgrade build
	python -m build
	python -m pip install --upgrade twine
	twine upload dist/*