# makefile

install:
	python3 -m pip install --upgrade pip &&\
		pip3 install -r requirements.txt &&\
		pip3 install -r bikeshare_model/requirements/requirements.txt &&\
		pip3 install -r bikeshare_model_api/requirements.txt

train:
	python3 bikeshare_model/train_pipeline.py --config config.yml

format:
	black *.py

lint:
	pylint --disable=R,C setup.py

test:
	python3 -m pytest tests/test_*.py

all : install train lint format