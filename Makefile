PY ?= python

.PHONY: test up down logs data registry probe unified validate report frames

test:
	$(PY) -m pytest

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api worker

# --- этап 2: данные ---
registry:
	$(PY) -m training.data.build_registry

probe:
	$(PY) -m training.data.probe

unified:
	$(PY) -m training.data.adapter_organizers

validate:
	$(PY) -m training.data.validate

report:
	$(PY) -m training.data.report

frames:
	$(PY) -m training.data.extract_frames

data: probe unified validate report
