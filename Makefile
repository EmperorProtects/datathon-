.PHONY: test up down logs

test:
	python -m pytest

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api worker
