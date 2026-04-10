run-docker-compose:
	docker compose up --build -d

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb