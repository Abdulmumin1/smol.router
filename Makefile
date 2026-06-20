.PHONY: test train demo example

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

train:
	PYTHONPATH=src python3 -m tiny_router train data/seed.jsonl --output model.json

demo: train
	PYTHONPATH=src python3 -m tiny_router route --model model.json "Prove this concurrent queue is linearizable"

example:
	PYTHONPATH=src python3 examples/benchmark_pipeline.py
