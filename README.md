# AquaGuard AI

**A checkpoint-based pressure–flow framework for diagnosing non-revenue water losses in distribution networks.**

AquaGuard AI is a research proof of concept for identifying four conditions—**normal operation, leakage, pipe burst, and unauthorized water withdrawal (theft)**—from simulated pressure–flow observations at ordered pipe checkpoints. The proposed decision-support layer also maps predicted incident types to suggested operator responses.

> **Research status:** The reported evaluation uses **synthetic data only**. AquaGuard AI has not been validated on operational utility telemetry, deployed as a field system, or shown to reduce real-world non-revenue water losses. Operator-response recommendations have not been operationally evaluated.

## Research scope

The simulation represents **12 illustrative Jordanian governorate configurations**, organized into **31 branches and 95 pipe segments**. A reading corresponds to one segment at one timestamp, with flow propagated between successive checkpoints in a branch.

The research pipeline derives **75 model features** from 44 source-data columns and investigates a calibrated stacked classifier built from LightGBM, XGBoost, a residual MLP, and a bidirectional LSTM. See the research manuscript and model implementation for the precise training and evaluation protocol; **the dataset generator does not retrain these models**.

## Repository contents

| Path | Purpose |
|---|---|
| [`aquaguard_v4_synth.py`](aquaguard_v4_synth.py) | Portable, verified v4 synthetic-data generator with optional hash verification |
| [`provenance/generate_v4_dataset_patched.py`](provenance/generate_v4_dataset_patched.py) | Archived recovered generator with the original pressure-clamp patches |
| [`jordan_v4_train.csv`](jordan_v4_train.csv) | Development dataset used for subsequent training and validation |
| [`jordan_v4_test.csv`](jordan_v4_test.csv) | Separate held-out synthetic test dataset |
| [`dataset_fingerprint.py`](dataset_fingerprint.py) | Dataset hash and statistical fingerprint utility |
| [`requirements-dataset.txt`](requirements-dataset.txt) | Tested dependencies for dataset regeneration |
| [`docs/DATASET_REPRODUCIBILITY.md`](docs/DATASET_REPRODUCIBILITY.md) | Detailed generator methodology, assumptions, audit evidence, and limitations |
| [`aquaguard_v4_2_model.py`](aquaguard_v4_2_model.py) | Model implementation supplied with the project |
| [`AquaGuard-AI.pbix`](AquaGuard-AI.pbix) | Power BI project file |

## Reproduce the synthetic datasets

From the repository root, use Python 3.13 and run:

```bash
python -m pip install -r requirements-dataset.txt
python aquaguard_v4_synth.py --out ./generated_data --verify
```

The generator writes **new CSVs to `./generated_data`**, leaving the committed datasets unchanged. Successful verification reports `MATCH repo file` for both files and exits with status code 0.

| Dataset | Records | Original columns | Simulated period | RNG seed |
|---|---:|---:|---|---:|
| `jordan_v4_train.csv` | 150,005 | 44 | 2019-01-01 00:00 to 2019-01-16 08:12 | 2024 |
| `jordan_v4_test.csv` | 30,020 | 44 | 2022-07-01 00:00 to 2022-07-04 01:30 | 2025 |

There are **1,579** development timestamps and **316** test timestamps, spaced **14 minutes** apart. Fault-injection probabilities are **20% and 5% per branch/timestamp**, respectively, with at most one fault-bearing segment per branch cycle. These produce **6.50% and 1.63% anomalous rows**, respectively; the injection probabilities are **not** row-level class percentages.

### Exact dataset fingerprints

| Dataset | SHA-256 |
|---|---|
| Development | `7eebeef1c1aab45953a771812a29642d054b9c89630d9e6b0f8c99615cce6307` |
| Test | `550322a1011670f3b56bb88ca94b2488b8c4df495ea60a3e38e60d765e2ff500` |

These values were verified against the existing repository CSVs. The regeneration was also tested with **Python 3.13.5, NumPy 2.3.5, and pandas 2.2.3**. Historical dependency versions from the original generation run were not recorded, so byte-for-byte agreement is demonstrated for tested environments rather than guaranteed for arbitrary future versions.

To inspect an existing CSV without regenerating it:

```bash
python dataset_fingerprint.py ./jordan_v4_train.csv
python dataset_fingerprint.py ./jordan_v4_test.csv
```

## Hydraulic and simulation assumptions

The generator computes `DP_Predicted` using the **Darcy–Weisbach head-loss relationship** with the **Churchill (1977) friction factor**. The implemented assumptions include:

- Flow in **L/min**, pressure in **PSI**, and pipe length in **metres**.
- A constant pipe diameter of **0.05 m**, absolute roughness of **0.26 mm**, and water kinematic viscosity of **1.004 × 10⁻⁶ m²/s**.
- Predicted pressure drop clipped to **0.01–60 PSI** and simulated downstream pressure bounded below by **0.5 PSI**.
- The recorded `HW_Coefficient` is **not used** in the generator's pressure-drop computation, despite being included as a dataset column.
- Fault signatures are prescribed by simulation rules, not measured from actual utility incidents.

The 60-PSI predicted-pressure cap affects **9.78% of development rows** and **22.74% of test rows**. The 0.5-PSI downstream-pressure floor affects **17.62%** and **30.09%**, respectively. These constraints may distort physically realistic pressure behavior and should be considered when interpreting model accuracy.

## Limitations and reproducibility boundary

Both datasets come from the **same simplified generator**, although their timestamps and random seeds differ. The short January 2019 and July 2022 windows do **not** establish multi-season or multi-year field generalization. Ramadan demand is not represented in these windows, and the Eid flag is randomly simulated rather than matched to actual holiday dates. Some governorate context fields are descriptive constants rather than inputs to flow/pressure generation. Sensor noise, drift, missing telemetry, complex network hydraulics, and diverse real fault behavior remain incompletely represented.

**Verified:** the recovered generator recreates both committed CSVs byte-for-byte, including their row counts, columns, and SHA-256 hashes.

**Not established by this check:** end-to-end reproduction of model-training metrics, probability-calibration quality, realistic hydraulic fidelity, generalization to unseen networks, or field accuracy. The scientific results must therefore be interpreted as **simulation-only proof-of-concept findings**.

For the full audit trail, hydraulic equations, network parameters, and caveats, see **[Dataset Generation and Reproducibility](docs/DATASET_REPRODUCIBILITY.md)**.
