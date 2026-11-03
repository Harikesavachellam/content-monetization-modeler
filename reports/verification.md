# Verification report

Checked on 3 November 2026, using Windows, Python 3.12 and the pinned project dependencies.

## Completed checks

| Area | Check and observed result |
| --- | --- |
| Automated regression tests | 44 passed with no warnings in the development environment and a separately installed environment. Covers source validation, duplicate handling, reproducible grouped splits, imputation, target exclusion, model persistence, invalid/missing inputs and all four Streamlit pages. |
| Dependencies | The independently installed environment passed dependency compatibility checks: 77 packages compatible. Direct versions are pinned in requirements.txt; the full build environment is recorded in requirements-lock.txt. |
| Notebook | All 13 code cells in the 28-cell notebook executed from a fresh kernel. No cell errors. The saved notebook contains the resulting outputs. Two complete executions reproduced the final test metrics. |
| Training | All five regressors trained successfully. Selection used validation data, with a predefined $0.05 RMSE tolerance for simpler candidates. Final fitting used training plus validation; test IDs remained separate. |
| Evaluation | Recomputed predictions match saved test predictions. Source/model hashes match metadata. Final MAE $3.1741, RMSE $13.6116 and R² 0.9516, across 23,857 test observations. |
| Python syntax | All delivered Python modules compiled successfully. |
| Streamlit logic | AppTest checks passed for all pages, valid/invalid predictions, optional missing inputs, filters, an incomplete date selection and a missing-model recovery message. |
| Real browser | Edge/Playwright opened all four pages, completed a prediction, checked the JSON download and checked the 120,000-row CSV download. No JavaScript errors were recorded. |
| Narrow display | A fresh 390 × 844 viewport had no horizontal document overflow. A screenshot is included. |
| Visualizations | Ten saved figures were inspected, along with desktop and narrow-screen application screenshots. |
| Documentation | The 76-page PDF was rendered and visually reviewed. Text extraction found no blank pages. It includes 1,040 source lines from the application, training, chart, utility, notebook and test code, with numbered statement explanations. |
| Packaged release | Built the ZIP, checked every member's CRC and compared every archived file with its source bytes. Extracted it into a fresh directory and ran the separate Python environment against that extracted project: all 44 tests passed in 8.51 seconds. After adding this report and its test XML, the final archive was rebuilt and checked again. Application, notebook, model and data files match the tested extraction. |

The XML test reports are saved as `reports/test_results.xml` and `reports/zip_test_results.xml`. Browser screenshots are in `docs/screenshots/`. Model scores and predictions are in this reports folder. A SHA-256 checksum file accompanies the final ZIP.

## Reproduce the checks

Extract the complete archive and install the pinned requirements as described in README.md. From the project folder, run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe tests/run_notebook.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The notebook reruns model training and replaces the generated model, metadata, charts and reports. The original dataset is preserved. To inspect the browser manually, visit each page, change filters, submit an estimate and try the CSV/JSON downloads.

## Practical limits

These checks found no outstanding failures in the tested environment. They cannot guarantee every future machine, package change or input will work. Windows and the tested local browser were exercised; macOS, Linux, public hosting, simultaneous-user load and exhaustive browser/accessibility audits were not verified. The Windows batch file's setup commands are documented; the installation and launch workflow was exercised through the Python command line.

The dataset is synthetic, with only 9,521–10,468 views per observation. The model rejects unsupported ranges. Real YouTube revenue accuracy is unverified. In the reserved test set, MAE was $0.61 with observed watch time but $51.67 when watch time was missing. This difference is shown in the app and learning guide.
