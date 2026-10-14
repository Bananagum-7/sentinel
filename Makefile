.PHONY: install run dashboard test eval gate load docker

install:
	pip install -r requirements.txt

run:
	uvicorn sentinel.app:app --reload

dashboard:
	streamlit run dashboard.py

test:
	pytest -q

eval:
	python eval/run_injection.py
	python eval/run_pii.py
	python eval/run_cache.py
	python eval/run_overhead.py

gate:
	python eval/gate.py

load:
	locust -f eval/locustfile.py --host http://localhost:8000 --users 20 --spawn-rate 5 --run-time 1m --headless

docker:
	docker compose up --build
