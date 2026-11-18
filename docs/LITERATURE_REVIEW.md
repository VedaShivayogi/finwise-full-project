# Week 1: Literature and repository review

**Author:** Veda Shivayogi Ramagaondanahalli, AI Intern, Cynaris Solutions
**Question this review answers:** *Which modelling, explanation and evaluation methods should an explainable credit-risk scorer for FinWise use, and why?*

Eleven sources were reviewed: **seven papers** and **four open-source repositories**. Sources 1-3 are the core references behind the three main design decisions
(model, explanation, evaluation). The summaries are in my own words and describe what I used from each source.

## Summary table

| # | Source | Type | Design decision it drove |
|---|---|---|---|
| 1 | Lessmann, Baesens, Seow & Thomas (2015), *Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research*, European Journal of Operational Research 247(1):124-136. doi:10.1016/j.ejor.2015.05.030 | Paper (core) | Compare against a logistic-regression baseline; report several accuracy measures, not one |
| 2 | Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*, KDD '16, pp. 785-794. arXiv:1603.02754 | Paper (core) | Use XGBoost as the primary model; regularisation and early stopping to control overfitting |
| 3 | Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS 30. arXiv:1705.07874 | Paper (core) | Use SHAP values for per-application explanations |
| 4 | Lundberg et al. (2020), *From local explanations to global understanding with explainable AI for trees*, Nature Machine Intelligence 2:56-67. doi:10.1038/s42256-019-0138-9 | Paper | Exact, fast TreeSHAP for the API; local-to-global importance |
| 5 | Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022 Datasets & Benchmarks. arXiv:2207.08815 | Paper | Justifies tree models over neural networks for a small tabular dataset |
| 6 | Hardt, Price & Srebro (2016), *Equality of Opportunity in Supervised Learning*, NeurIPS 29. arXiv:1610.02413; and Mitchell et al. (2019), *Model Cards for Model Reporting*, FAT* '19. arXiv:1810.03993 | Papers | Fairness metrics (error-rate parity) and the structure of the model card |
| 7 | `dmlc/xgboost` (github.com/dmlc/xgboost) | Repo | Training library; native `pred_contribs` used for fast SHAP in the API |
| 8 | `shap/shap` (github.com/shap/shap) | Repo | Reference TreeExplainer; used to verify the native SHAP values |
| 9 | `mlflow/mlflow` (github.com/mlflow/mlflow) | Repo | Experiment tracking, model registry-style run ids for versioning |
| 10 | `fairlearn/fairlearn` (github.com/fairlearn/fairlearn) | Repo | Reference definitions for group-fairness metrics (I implemented the audit with plain pandas) |

## Core sources in detail

### 1. Lessmann et al. (2015): credit-scoring benchmark
**What it is.** A large benchmark of **41 classifiers on eight real credit-scoring datasets**, updating a well-known 2003 study. It also compares **six accuracy
indicators** and assesses the **financial impact** of better scorecards.
**What I took from it.**
- *Always keep a simple baseline.* Logistic regression is the long-standing industry scorecard method, so every complex model must be compared against it.
  My project does this (baseline test Gini 0.849 versus champion 0.817, reported honestly in the model card).
- *Use more than one metric.* Different indicators can rank models differently, so I report Gini/AUC, KS and Brier score and the default rate per risk band.
- *Accuracy only matters if it is useful to the business*, which led to the low/medium/high band table instead of a single score.

### 2. Chen & Guestrin (2016): XGBoost
**What it is.** The paper behind XGBoost: gradient-boosted decision trees with a **regularised objective**, sparsity-aware split finding and an engineering design
for speed and scale.
**What I took from it.**
- Gradient-boosted trees handle mixed numeric/categorical financial features, non-linear effects and interactions (for example high debt *and* low credit score) without manual feature crossing.
- The regularisation terms (`gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`) and subsampling are the main tools against overfitting on a small dataset; I tuned three configurations and used
  **early stopping on a validation set** (best iteration 15).
- `scale_pos_weight` handles the 23.8% default rate (set to about 3.2, the negative-to-positive ratio). Side effect noted in the model card: raw scores are not calibrated probabilities.

### 3. Lundberg & Lee (2017): SHAP
**What it is.** A framework that unifies several explanation methods as *additive feature attribution* and shows that **Shapley values** are the one attribution satisfying
desirable properties (local accuracy, missingness, consistency).
**What I took from it.**
- Every prediction is decomposed as *average prediction + sum of per-feature contributions*. That is exactly the **waterfall chart** the client asked for, and the contributions are additive by construction
  (tested: contributions sum to the model output).
- The top-3 risk factors shown in the UI are the three largest positive SHAP contributions for that applicant.

## Supporting sources (short)
- **4. Lundberg et al. (2020)** gives an exact, polynomial-time algorithm (**TreeSHAP**) for tree ensembles and shows how many local explanations combine into global importance. This is why per-request explanations are fast enough for an API (about 2.6 ms CPU per request in my load test).
- **5. Grinsztajn et al. (2022)** benchmarks tree-based and deep models on 45 tabular datasets and finds that **tree-based models remain state-of-the-art on medium-sized data (around 10,000 samples) even before counting their speed**. My data has 1,000 rows, so neural networks were not considered.
- **6. Hardt et al. (2016)** defines *equality of opportunity* (similar true-positive rates across groups), which motivated reporting "defaulters caught" per age group beside approval-rate impact ratios.
  **Mitchell et al. (2019)** proposed model cards; `MODEL_CARD.md` follows its sections (intended use, data, metrics, ethical considerations, caveats).
- **7-10. Repositories.** `xgboost` and `shap` are used directly. `mlflow` tracks every experiment run (parameters, metrics, artifacts). `fairlearn` was reviewed for metric definitions (selection rate, equal-opportunity difference); to keep the dependency list small, the audit in `scripts/bias_audit.py` is written with pandas.

## What the literature changed in my plan
| Initial idea | After the review |
|---|---|
| Train XGBoost and report AUC | Add a logistic-regression baseline and report Gini, AUC, KS, Brier and per-band default rates |
| Explain with feature importance | Use per-application SHAP waterfalls, built on the same trees as the score |
| One fairness number | Report impact ratio *and* defaulter catch rate by group; state clearly what could not be audited |
| README only | Write a model card as the main documentation artifact |

## Gaps and limits of this review
- The benchmark papers use much larger, real datasets. FinWise's data has 1,000 rows and looks synthetic, so published rankings may not transfer.
- Regulatory guidance (for example RBI rules on fair lending and digital lending, or adverse-action requirements) was **not** reviewed; it is listed as a required step before real use in `MODEL_CARD.md`.
- Calibration methods (Platt scaling, isotonic regression) and credit-specific scorecard methods (weight-of-evidence binning) are noted as future work.

## References
1. Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124-136.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785-794.
3. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
4. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56-67.
5. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022, Datasets and Benchmarks Track*.
6. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems 29*.
7. Mitchell, M., et al. (2019). Model cards for model reporting. *Proceedings of FAT\* 2019*.
