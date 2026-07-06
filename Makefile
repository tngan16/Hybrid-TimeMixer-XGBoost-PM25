.PHONY: install validate clean baselines experiment multiseed figures app test paper full quick
install:
	pip install -e ".[dev,app]"
validate:
	python scripts/validate_environment.py
clean:
	python scripts/run_cleaning.py
baselines:
	python scripts/run_baselines.py
experiment:
	python scripts/run_experiment.py --epochs 30
multiseed:
	python scripts/run_multiseed.py --epochs 30
figures:
	python scripts/make_figures.py
full:
	python scripts/run_full_study.py --epochs 30
quick:
	python scripts/run_full_study.py --epochs 3
app:
	streamlit run app/streamlit_app.py
test:
	pytest
paper:
	cd paper && xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
