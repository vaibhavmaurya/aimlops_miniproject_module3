# makefile

install:
	python -m pip install --upgrade pip &&\
		pip install -r requirements.txt

train:
	python bikeshare_model/train_pipeline.py

format:
	black *.py

lint:
	pylint --disable=R,C setup.py

test:
	python -m pytest tests/test_*.py

all : install train lint test format