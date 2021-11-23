black:
	black --line-length 80 --diff --color examples kfac && \
	black --line-length 80 examples kfac

flake8:
	flake8 . --count --show-source --statistics

pytest:
	pytest --cache-clear kfac

test: black flake8 pytest
