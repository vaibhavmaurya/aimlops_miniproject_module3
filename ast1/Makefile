j_install:
	pip3 install -r requirements.txt

j_format:
	black script.py

	
j_lint:
	pylint script.py 
	 
j_test:
	python3 -m pytest 
	
all: j_install j_format j_lint j_test