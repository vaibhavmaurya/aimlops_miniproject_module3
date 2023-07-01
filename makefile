# makefile

install:
	python -m pip install --upgrade pip &&\
		pip3 install -r requirements.txt &&\
		pip3 install -r bikeshare_model/requirements/requirements.txt

train:
	python3 bikeshare_model/train_pipeline.py --config bikeshare_model_api/config/config.yml

format:
	black *.py

lint:
	pylint --disable=R,C setup.py

test:
	python3 -m pytest tests/test_*.py

all : install train lint test format