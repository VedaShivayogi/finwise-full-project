# 5-minute live demo: script and checklist

## Before you press record (10 minutes)
- [ ] `python -m pytest -q` shows **41 passed** on your PC (if not, fix before recording).
- [ ] Terminal A: `python -m uvicorn serving.app:app --port 8000` is running. Browser tab 1: http://localhost:8000
- [ ] Browser tab 2: http://localhost:8000/docs. Tab 3: your GitHub repo **Actions** page showing a green tick. Tab 4: `MODEL_CARD.md` on GitHub.
- [ ] Terminal B (empty) in the project folder, font size large. Close notifications, chat apps and unrelated tabs.
- [ ] Record with Windows Game Bar (Win+G) or OBS, 1080p, microphone test first. Browser zoom 110%.
- [ ] After recording: upload as **Unlisted** on YouTube (or Google Drive, "Anyone with the link"), then open the link in a private window to check it plays.

## Timeline (say the quoted lines in your own words)
| Time | Show | Say |
|---|---|---|
| 0:00-0:30 | Title / UI home | "I'm Veda, AI intern at Cynaris Solutions. FinWise needs to score loan applicants and explain every score. This is the finished system: model, API, UI, tests and CI." |
| 0:30-1:30 | UI: click **Overleveraged**, **Score applicant** | "High risk, score 457. The top 3 risk factors are debt-to-income, the high-DTI flag and credit score. The chart is a SHAP waterfall: red bars push risk up, green down, starting from the average applicant." Then click **Steady salaried**: "Low risk, and the factors are tiny, so the UI marks them minor." |
| 1:30-2:15 | UI: **Model & fairness** tab | "Gini 0.817 on unseen test data, target was 0.45. Bands: low risk never defaulted, high risk defaulted 66%. Age audit passes the four-fifths rule, lowest 0.90. Gender is not audited because the data has no gender column, and I say that openly." |
| 2:15-3:00 | Tab 2: `/docs` -> **POST /predict** -> Try it out -> Execute. Then set `credit_score` to 100 and Execute again | "The same model is a REST API. A valid request returns score, band and explanations. An invalid value returns 422 and is never scored." |
| 3:00-3:45 | Terminal B: `python -m pytest -q -m "not live"`, then tab 3 (green Actions run) | "41 tests, including one that sends the held-out test rows through the API and reproduces the training Gini. Every push runs these tests, a fairness gate and the front-end build in GitHub Actions. It's green." |
| 3:45-4:20 | Open `docs/WEEK4_SERVING_AND_LOAD_TEST_REPORT.md` results table | "With 50 requests at once: zero errors, about 366 requests per second and p95 under 230 ms on one shared CPU core. A fast scoring path is about 9 times cheaper than my first version, proven identical by tests." |
| 4:20-5:00 | Tab 4: model card (limitations section) | "The model card lists the data, metrics, bias audit and deployment steps. Honest limits: the data looks synthetic, scores are not calibrated, age is a model input, and gender wasn't auditable. Next steps are real data, calibration and a gender audit. Thank you." |

## If something goes wrong on camera
- Server not responding: Terminal A closed. Restart it and refresh; keep talking.
- Page blank: hard refresh (Ctrl+F5).
- Tests slow: use `-m "not live"` (about 10 s) rather than the full suite.
