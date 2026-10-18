"""Ключ Redis, который воркер выставляет после прогрева моделей (для /readyz)."""
READY_KEY = "cashguard:models_ready"
