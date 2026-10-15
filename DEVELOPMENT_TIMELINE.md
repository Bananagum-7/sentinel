# Development Timeline

> **These dates are estimates, not verified history.**
> The "Estimated date (2026)" column records the author's recollection of when each
> milestone was developed locally between July 15 and October 14, 2026, before the
> code was committed. Nothing in the repository verifies those dates: every
> original commit was created on 2026-10-14 between 21:45 and 21:55 (+0530).
> The "Git timestamp" column is the real commit date. The reconstructed commits
> keep the timestamps of the original commits they were split from.

The `main` branch holds the original 7 commits. The `reconstructed-history` branch
splits them into smaller commits. Every commit there contains only code from the
original commits, and the tree at the README commit is identical to `main`.


## Phase 1: Foundation (July 15–31, 2026)

Project setup, environment configuration, per-app policies, gateway key authentication and rate limiting.

| # | Estimated date (2026) | Commit | Change | Git timestamp |
|---|---|---|---|---|
| 1 | Jul 15 | `555dc12` | chore: initialize project with dependencies and environment template | 2026-10-14 21:45:21 +0530 |
| 2 | Jul 18 | `8db8968` | feat(policy): add per-app policy schema with YAML defaults merging | 2026-10-14 21:45:21 +0530 |
| 3 | Jul 22 | `c9f7789` | feat(auth): resolve gateway keys from env vars with constant-time comparison | 2026-10-14 21:45:21 +0530 |
| 4 | Jul 26 | `080f18a` | feat(ratelimit): add per-app token-bucket rate limiter with Retry-After | 2026-10-14 21:45:21 +0530 |
| 5 | Jul 30 | `c3678d4` | test: cover policy merging, key auth and token-bucket refill | 2026-10-14 21:45:21 +0530 |

## Phase 2: Security guards and upstream access (August 2026)

PII detection and redaction, prompt-injection detection, the provider client and the fallback router.

| # | Estimated date (2026) | Commit | Change | Git timestamp |
|---|---|---|---|---|
| 6 | Aug 4 | `786bd76` | feat(pii): add regex PII detector with Luhn-validated card numbers | 2026-10-14 21:46:49 +0530 |
| 7 | Aug 8 | `8d6815c` | feat(pii): add per-request placeholder redaction and restoration | 2026-10-14 21:46:49 +0530 |
| 8 | Aug 12 | `5017d5a` | feat(pii): add optional Presidio NER detector for names and locations | 2026-10-14 21:46:49 +0530 |
| 9 | Aug 16 | `763fa8e` | feat(injection): add weighted heuristic prompt-injection detector | 2026-10-14 21:46:49 +0530 |
| 10 | Aug 20 | `fb32c1b` | feat(injection): add optional DeBERTa classifier and max-score combiner | 2026-10-14 21:46:49 +0530 |
| 11 | Aug 23 | `03d6d10` | test(guards): cover PII detection, redaction round-trip and injection rules | 2026-10-14 21:46:49 +0530 |
| 12 | Aug 27 | `7827d10` | feat(providers): add OpenAI-compatible provider client with error classification | 2026-10-14 21:49:40 +0530 |
| 13 | Aug 31 | `c2085bd` | feat(router): add model fallback chain with retry and exponential backoff | 2026-10-14 21:49:40 +0530 |

## Phase 3: Caching, observability, gateway and API (September 2026)

Semantic cache, request log and tracing, the guarded request pipeline, streaming, the OpenAI-compatible API and end-to-end tests.

| # | Estimated date (2026) | Commit | Change | Git timestamp |
|---|---|---|---|---|
| 14 | Sep 4 | `8625f9b` | feat(cache): add exact and semantic response cache with critical-token check | 2026-10-14 21:49:40 +0530 |
| 15 | Sep 8 | `06edcb8` | test: add scripted FakeProvider and cover cache and router failure modes | 2026-10-14 21:49:40 +0530 |
| 16 | Sep 12 | `e6668ef` | feat(observability): add metadata-only SQLite request log and OTel tracing | 2026-10-14 21:49:40 +0530 |
| 17 | Sep 16 | `acc5386` | feat(gateway): add guarded request pipeline with output PII guard | 2026-10-14 21:49:40 +0530 |
| 18 | Sep 20 | `2a6b821` | feat(gateway): add SSE streaming with split-placeholder restoration | 2026-10-14 21:49:40 +0530 |
| 19 | Sep 24 | `136373f` | feat(api): expose OpenAI-compatible endpoints with key auth and metrics | 2026-10-14 21:49:40 +0530 |
| 20 | Sep 28 | `0c6758b` | test(api): add end-to-end tests for auth, PII, injection, cache, limits and streaming | 2026-10-14 21:49:40 +0530 |

## Phase 4: Evaluation, hardening and release (October 1–14, 2026)

Guard evals and the CI regression gate, guard tuning driven by the first eval run, dashboard, Docker, CI and documentation.

| # | Estimated date (2026) | Commit | Change | Git timestamp |
|---|---|---|---|---|
| 21 | Oct 1 | `0060529` | test(eval): add eval helpers and prompt-injection eval | 2026-10-14 21:52:22 +0530 |
| 22 | Oct 2 | `fa6ad55` | test(eval): add Faker-based PII redaction eval | 2026-10-14 21:52:22 +0530 |
| 23 | Oct 3 | `e0adb3b` | test(eval): add semantic cache paraphrase and near-miss eval | 2026-10-14 21:52:22 +0530 |
| 24 | Oct 5 | `3126328` | perf(eval): add gateway overhead benchmark and Locust load test | 2026-10-14 21:52:22 +0530 |
| 25 | Oct 6 | `b1e72ac` | test(eval): add guard regression gate with CI thresholds | 2026-10-14 21:52:22 +0530 |
| 26 | Oct 7 | `c7552d1` | fix(injection): normalise leetspeak and spaced-letter obfuscation | 2026-10-14 21:53:50 +0530 |
| 27 | Oct 8 | `a7a4066` | fix(injection): catch paraphrased overrides and prompt extraction, split weak jailbreak signals | 2026-10-14 21:53:50 +0530 |
| 28 | Oct 9 | `809580c` | fix(pii): treat bare 10-digit numbers as phones only when mobile-like | 2026-10-14 21:53:50 +0530 |
| 29 | Oct 10 | `18c4b41` | fix(cache): replace critical-token match with content-word intent guard | 2026-10-14 21:53:50 +0530 |
| 30 | Oct 11 | `9a432bf` | perf(cache): lower semantic similarity threshold to 0.5 based on eval | 2026-10-14 21:55:06 +0530 |
| 31 | Oct 12 | `0d3a333` | feat(app): add fake-upstream mode for demos and load tests | 2026-10-14 21:55:06 +0530 |
| 32 | Oct 12 | `83fed98` | feat(dashboard): add Streamlit dashboard over the request log | 2026-10-14 21:55:06 +0530 |
| 33 | Oct 13 | `00146e4` | chore: add Docker, Makefile and CI workflow running tests and guard gate | 2026-10-14 21:55:06 +0530 |
| 34 | Oct 14 | `1e8a8ad` | docs: add README with architecture, design decisions and eval results | 2026-10-14 21:55:06 +0530 |

## Post-release fixes (real dates, not estimates)

| # | Date | Commit | Change | Git timestamp |
|---|---|---|---|---|
| 35 | Oct 15 | `00ad105` | fix(eval): track evaluation datasets required for reproducible CI | 2026-10-15 20:22:48 +0530 |
| 36 | Oct 15 | this commit | docs: add development timeline with estimated milestone dates | see `git log` |

## Notes

- The eval-driven tuning in commits 25–29 is real history. When the regression
  gate was first run it failed (injection recall 0.70, false-positive rate 0.10,
  1 PII false positive, 14 wrong cache hits). Each `fix` commit closes one of
  those failures, and all gates pass from commit 29 onward.
- Tests pass at every commit (3 tests at the first test commit, 47 at the end).
