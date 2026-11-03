# Content Monetization Modeler

A beginner-friendly Python project that estimates YouTube advertising revenue for one reporting observation. It includes a complete analysis notebook, five regression models, interactive Streamlit charts and a detailed PDF explanation.

The supplied data is synthetic. The model demonstrates a learning workflow; it is not a verified estimate of real YouTube payments.

## Start here

1. Extract the entire ZIP. Keep its folders together.
2. Install **64-bit Python 3.12** from [python.org](https://www.python.org/downloads/), if needed.
3. On Windows, double-click **START_APP.bat**. It creates a local virtual environment, installs the pinned packages and launches the app. The first setup needs Internet access.
4. Alternatively, follow the terminal instructions below. The saved model is included, so retraining is optional for running the app.
5. Read `docs/project_explanation.pdf` and then open `notebooks/project_analysis.ipynb` in VS Code or Jupyter.

## Terminal setup

Open a terminal **inside this project folder**.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS/Linux (instructions provided; these operating systems were not used for verification):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. Keep the terminal running. Press Ctrl+C to stop the server. Virtual-environment activation is optional because the commands above call its Python explicitly.

If Python is installed as `python` rather than `py`, use `python -m venv .venv` after confirming `python --version` reports 3.12. Keep the tested scikit-learn, NumPy, Pandas and joblib versions when loading the supplied model. The app checks them and provides a recovery message if they differ.

`requirements.txt` pins the direct project dependencies. `requirements-lock.txt` records the complete tested environment, including documentation libraries. The direct requirements are sufficient for the project. The lock file is useful for reproducing the exact tested environment.

## What is predicted?

Target: `ad_revenue_usd`, in US dollars for the supplied reporting observation.

Inputs: views, likes, comments, total watch time in minutes, video length in minutes, subscribers, category, device and country. All performance totals must refer to the same reporting period. The app has no YouTube API connection. For a future scenario, you must supply assumed performance figures; the app does not forecast future views or watch time.

The app has four pages:

| Page | Purpose |
| --- | --- |
| Overview | Project introduction, sample statistics and limitations |
| Explore Data | Category, device, country and date filters; distributions, box plots, correlations, date summaries and CSV download |
| Predict Revenue | Validated input form, revenue estimate, missing-input notes and JSON download |
| Model Results | Five-model validation comparison, final test metrics, missing-watch-time errors and importance plots |

## Data and preparation

- Source: [provided CSV](https://drive.google.com/file/d/1joozv6lXGgvnYZxbdKlV4VpVp2rbR_7s/view), linked from the [project brief](https://docs.google.com/document/d/1zBCNNywwsDs3oEquJ7Q6Dwdt87_B5E-LVdzmjkOZOvw/edit).
- Original shape: 122,400 rows, 12 columns. Exact duplicates removed: 2,400. Remaining observations: 120,000.
- After deduplication, likes, comments and watch time each contain 6,000 missing values (5%).
- Observed dates: 9 June 2024–8 June 2025. The source contains six categories, six countries and four devices.
- Repeated IDs are not discarded. They can carry distinct records, and their inconsistent categories and video lengths are documented as synthetic-data limitations.
- Numeric medians, missing indicators, categorical imputation, one-hot encoding and scaling are fitted inside the pipeline. They use only training data during comparison.
- Features: engagement rate `(likes + comments) / views` and average watch minutes `watch_time_minutes / views`.
- Ratios are created before imputation. A missing ratio receives its own median. It need not equal a ratio calculated from separately imputed raw fields.
- The target, video ID and date are excluded from predictors. No revenue-derived feature is used.

The original CSV stays unchanged. `cleaned_youtube_data.csv` is an unscaled analysis export with missing flags and partition labels, filled using medians fitted on training + validation. **Always train from the original CSV**, so missingness and the evaluation split remain correct. The app's EDA uses deduplicated source values before imputation. Notebook model-development EDA uses training rows.

## Model evaluation

Split by video ID with seed 42:

| Partition | Video IDs | Rows |
| --- | ---: | ---: |
| Training | 3,000 | 72,176 |
| Validation | 1,000 | 23,967 |
| Test | 1,000 | 23,857 |

The same ID never appears in two partitions. Because identifiers are synthetic and inconsistent, this does not establish generalization to real channels or future dates.

| Candidate | Validation MAE (USD) | Validation RMSE (USD) | Validation R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 3.2371 | 13.8101 | 0.9506 |
| Ridge Regression | 3.2890 | 13.8082 | 0.9506 |
| Decision Tree | 5.2908 | 14.1681 | 0.9480 |
| Random Forest | 3.6774 | 13.8766 | 0.9501 |
| Gradient Boosting | 4.1122 | 13.9009 | 0.9499 |

**Selected: Linear Regression.** Ridge has the lowest validation RMSE by about $0.0019. The predefined rule prefers the simpler model within $0.05 of the lowest RMSE. Linear Regression also has lower validation MAE and meets the brief's baseline requirement. Settings are fixed and modest; exhaustive tuning is not claimed.

The selected model was refitted on training + validation and then evaluated on the reserved test set:

- **MAE: $3.1741**
- **RMSE: $13.6116**
- **R²: 0.9516**
- Mean-only reference MAE: $53.2698.

| Test group | Rows | MAE (USD) | RMSE (USD) |
| --- | ---: | ---: | ---: |
| Watch time observed | 22,660 | 0.6121 | 2.0283 |
| Watch time missing | 1,197 | 51.6736 | 60.1231 |

R² is not an accuracy percentage. MAE and RMSE are in dollars. Test metrics describe these records, not a confidence interval for an individual estimate. The missing-watch-time group explains why RMSE is much greater than MAE.

Permutation importance uses the training-fitted selected model on 1,500 validation rows, before the final refit, with three shuffles per original input. Importance measures predictive reliance, not causation. Correlated predictors can share information.

## Project structure

```text
content-monetization-modeler/
  app.py                        Streamlit interface
  utils.py                      Shared validation, features and pipelines
  train_model.py                Five-model training and evaluation
  charts.py                     Ten saved analysis and diagnostic figures
  START_APP.bat                 Windows setup and launch helper
  requirements.txt              Pinned direct dependencies
  requirements-lock.txt         Complete tested environment
  .gitignore                   Excludes local environments and caches
  .streamlit/config.toml         Colors and local app configuration
  notebooks/project_analysis.ipynb
  data/youtube_ad_revenue_dataset.csv
  data/cleaned_youtube_data.csv
  models/revenue_pipeline.joblib
  models/metadata.json           Input ranges, versions, split and hashes
  reports/                      Metrics, predictions, partitions and figures
  docs/project_explanation.pdf  Detailed learning and code guide
  docs/screenshots/              Captures of the working application
  tests/test_project.py          Automated regression checks
  tests/run_notebook.py          Execute the notebook from a fresh kernel
```

The application needs `utils.py`, the model, metadata, original dataset and reports together. The whole folder is the deliverable; copying only `app.py` is insufficient.

## Retrain and verify

Windows commands from the project folder:

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe tests/run_notebook.py
```

On macOS/Linux replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

Training overwrites the generated model, metadata, cleaned-data export, reports and figures. It preserves the original CSV. The notebook also retrains all five models and overwrites these generated outputs. Close other training processes while running either workflow. Training took approximately 40 seconds in notebook verification on the tested machine; other machines may take longer.

For interactive notebook use, choose the project's `.venv` Python kernel in VS Code. The notebook runner automatically uses the Python interpreter that launches it and does not require a globally installed kernel.

See `reports/verification.md` for the completed checks and their practical limits.

## Input limits and troubleshooting

| Situation | What to do |
| --- | --- |
| Python not found | Install 64-bit Python 3.12 and reopen the terminal |
| Missing Python package | Run the requirements installation command using the same Python that launches the app |
| Saved model unavailable or versions differ | Install pinned requirements; run `train_model.py` if artifacts need rebuilding |
| Missing dataset | Restore `data/youtube_ad_revenue_dataset.csv` from the ZIP or supplied source |
| Missing figure/report | Run `train_model.py` from the project folder |
| Port already in use | Add `--server.port 8502` to the Streamlit command |
| No matching observations | Widen the date range or clear filters |
| Input outside the fitted range | Use a supported value; the learning model deliberately rejects extrapolation |
| Unknown likes/comments/watch time | Mark it unknown in the form rather than entering zero |
| Notebook cannot find `utils.py` | Keep the notebook inside the supplied `notebooks` folder |

The view range is only 9,521–10,468. Float input limits are rounded inward to two decimals so values shown as limits are accepted. Valid marginal ranges do not guarantee that every combination is representative. Missing watch time materially reduces prediction quality.

An average watch time above video length occurs in the supplied data; the app shows a note rather than automatically deleting such a record. This ratio is not presented as a verified retention percentage.

Joblib model files use pickle-based serialization. Load only trusted model files. There is no arbitrary model-upload feature. The local app does not store personal user accounts, payment information or submitted prediction histories.

## GitHub and presentation

When you are ready to submit, create a repository, add this project, review the files, commit and push using Git or GitHub Desktop. Keep `.venv` and caches out of Git. The original dataset download link is recorded above for attribution and reproducibility.

The submission guide requests your repository URL in the Zen portal's Backend Source Code field and the project booking form, with public visibility or reviewer access. No repository was published and no booking was submitted by this project build.

The PDF includes a 12-minute explanation outline and viva questions. Be prepared to explain why this is regression, how missing data is handled, why IDs are separated, how the five models were compared, why Linear Regression was selected, and why real-world generalization remains unverified.
