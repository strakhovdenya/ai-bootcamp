run-docker-compose:
	docker compose up --build -d

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

run-evals-retriever:
	uv sync
	PYTHONPATH="$(PWD)/apps/api;$(PWD)/apps/api/src;$$PYTHONPATH;$(PWD)" uv run --env-file .env python -m evals.eval_retriever

run-evals-dataset:
	uv sync
	PYTHONPATH="$(PWD)/apps/api;$(PWD)/apps/api/src;$$PYTHONPATH;$(PWD)" uv run --env-file .env python -m evals.analyze_dataset_quality_html \
		--dataset-name "rag-evaluation-dataset-30-v2" \
		--expected-single 15 \
		--expected-multi 10 \
		--expected-cannot-answer 5