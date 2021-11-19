black:
	black --line-length 80 examples kfac

flake8:
	flake8 . --count --show-source --statistics

pytest:
	pytest --cache-clear --cov=kfac --cov-report term-missing kfac

test: black flake8 pytest
