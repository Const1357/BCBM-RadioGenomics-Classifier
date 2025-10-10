ENV_NAME = BCBM_classifier_env

.PHONY: env install clean lint test data preprocess train eval

setup:
	make env
	make install
	make data
# 	make preprocess

clean:
	@echo ">>> Cleaning up"
	rm -rf $(ENV_NAME)
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf logs/*

env:
	@echo ">>> Creating virtual environment: $(ENV_NAME)"
	python3 -m venv $(ENV_NAME)
	@echo ">>> IMPORTANT: Activate with: source $(ENV_NAME)/bin/activate"

install:
	@echo ">>> Installing dependencies"
	pip install --upgrade pip
	pip install -r requirements.txt
	chmod +x scripts/*
	chmod +x scripts/runners/*
	source ~/.bashrc

preprocess:
	@echo ">>> Preprocessing data"
	python3 src/utils/process_data.py



train:
	@echo ">>> Training model (CNN)"
	./scripts/runners/run_CNN.sh

eval:
	@echo ">>> Evaluating model"
	python3 src/evaluate_model.py
