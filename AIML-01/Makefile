.PHONY: up down health logs clean lint test k8s-up k8s-down demo

# Start all services
up:
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@powershell -Command "Start-Sleep -Seconds 15"
	$(MAKE) health

# Stop all services
down:
	docker compose down

# Remove volumes too
clean:
	docker compose down -v

# Check health of all services
health:
	@echo "=== Health Checks ==="
	@echo "--- Kafka ---"
	@docker exec fraud-kafka kafka-topics --bootstrap-server localhost:9092 --list > NUL 2>&1 && echo "OK" || echo "FAIL"
	@echo "--- Redis ---"
	@docker exec fraud-redis redis-cli ping 2> NUL || echo "FAIL"
	@echo "--- Postgres ---"
	@docker exec fraud-postgres pg_isready -U fraud 2> NUL || echo "FAIL"
	@echo "--- MLflow ---"
	@curl -sf http://localhost:5000/health > NUL 2>&1 && echo "OK" || echo "FAIL"
	@echo "=== Done ==="

# View logs
logs:
	docker compose logs -f

# Lint
lint:
	python -m ruff check src/ tests/

# Run tests
test:
	python -m pytest tests/ -v

# Format
fmt:
	python -m ruff format src/ tests/

k8s-up:
	powershell -ExecutionPolicy Bypass -File scripts/k8s_up.ps1

k8s-down:
	powershell -NoProfile -Command "if (Test-Path data/tools/minikube.exe) { & ./data/tools/minikube.exe stop -p fraud-demo } else { minikube stop -p fraud-demo }"

demo:
	python -X utf8 -m scripts.verify_pipeline
