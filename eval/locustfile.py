"""HTTP load test against a running gateway.

    SENTINEL_KEY=... locust -f eval/locustfile.py --host http://localhost:8000 \
        --users 20 --spawn-rate 5 --run-time 1m --headless

Start the server with SENTINEL_FAKE_UPSTREAM=1 so the test measures the
gateway (fixed 300 ms fake upstream), not Groq's free-tier rate limits.
"""

import os
import random

from locust import HttpUser, between, task

QUESTIONS = [
    "How do I reset my password?",
    "Where is my order? My email is test{n}@example.com",
    "What is your refund policy?",
    "Summarize: customer cannot log in after changing phone number {n}",
]


class GatewayUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.headers = {"Authorization": f"Bearer {os.environ['SENTINEL_KEY']}"}

    @task
    def chat(self):
        q = random.choice(QUESTIONS).format(n=random.randint(1, 10_000))
        self.client.post("/v1/chat/completions", headers=self.headers, name="/v1/chat/completions",
                         json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": q}]})
