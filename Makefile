.PHONY: test check train demo example

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

check: test
	PYTHONPATH=src python3 -m compileall -q src tests examples
	git diff --check

train:
	PYTHONPATH=src python3 -m smol_router train data/seed.jsonl --output model.json

demo: train
	PYTHONPATH=src python3 -m smol_router route --model model.json "Prove this concurrent queue is linearizable"

example:
	PYTHONPATH=src python3 examples/benchmark_pipeline.py
