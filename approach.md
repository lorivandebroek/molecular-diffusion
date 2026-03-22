# Approach: SMILES + temperature → diffusion coefficient (in water)

This document describes the end-to-end process from the supplemental Excel file to a trained regression model. The scientific goal is to predict the **aqueous diffusion coefficient** \(D\) from **molecular structure (SMILES)** and **temperature**, for use at inference time on new molecules within a documented validity range.

---

## 1. Data source

- **File:** `jp5c01881_si_002.xlsx` (supporting information for Han et al., multimodal learning of diffusion in water).
- **Sheet for supervised learning:** **`Table S2`** (“The dataset of diffusion coefficients”).
- **Columns to use:**
  - **`SMILSES`** — SMILES string (column name is a typo in the workbook).
  - **`Temperature`** — in kelvin (observed range in this extract: roughly 273.8–394 K).
  - **`D`** — diffusion coefficient (confirm **units** in the main paper; typically **cm²/s** in this literature).

**Auxiliary columns** (`NO`, `Chem_NO`, `literature`, `FORMULA`, `NAME`, `CAS No`) are useful for traceability, deduplication checks, and debugging; they are **not** required as model inputs unless you explicitly want source-paper effects (usually avoided).

**Other sheets:** `Table S1` is bibliography only. `Tables S3–S5` are summary metrics from other models. `Table S6` is predicted vs experimental \(D\) for the authors’ model—useful as an **optional external reference**, not as the primary training table. `Tables S7–S8` are feature-importance summaries from their models.

**Scale (approximate):** on the order of **~20k** labeled rows and **~7k** unique SMILES, with **multiple temperatures per compound** for many molecules—enough for fingerprint-based networks and reasonable graph or pretrained-encoder setups if splits and evaluation are done carefully.

---

## 2. Cleaning and standardization

1. **Parse SMILES** with a chemistry toolkit (e.g. RDKit).
2. **Canonicalize** SMILES so the same structure always maps to one string.
3. **Remove or fix** invalid or empty SMILES; record counts.
4. **Deduplicate** rows that share the same **(canonical SMILES, temperature)**. The workbook contains a modest number of duplicate pairs—either **drop** extras or **average** \(D\) and keep one row; document the rule.
5. **Confirm** \(D\) units once from the paper and keep all training/inference code consistent.

---

## 3. Target engineering

- \(D\) often spans **many orders of magnitude**. Training with plain MSE on \(D\) can overweight large values.
- **Recommended:** predict **`log10(D)`** or **`ln(D)`**, optimize loss in log space, then transform back to \(D\) for reporting and application.
- Report metrics in **log space** (primary for optimization) and, if useful, on **linear \(D\)** after back-transformation.

---

## 4. Input scaling

- Keep temperature in **kelvin** (physically consistent with the data).
- **Normalize** temperature (and any continuous descriptors) using **statistics computed on the training set only** (e.g. z-score or min–max), then apply the same transform at validation, test, and inference.

---

## 5. Molecular representation

Choose by effort and desired accuracy:

| Approach | Notes |
|----------|--------|
| **Morgan / ECFP fingerprints + MLP** | Fast baseline; concatenate normalized **\(T\)**. |
| **Graph neural network (GNN)** | Strong inductive bias for structure; fuse graph embedding with **\(T\)**. |
| **Pretrained chemistry Transformer (SMILES or graph) + regression head** | Often strong with tuning; fine-tune with **\(T\)** in the head or via fusion. |

Start with a **strong baseline** (fingerprints + MLP on `log(D)`) before investing in heavier architectures.

---

## 6. Data splitting (critical)

**Always split by molecule, not by row:** every row for a given canonical SMILES must fall entirely in **train**, **validation**, or **test**. Otherwise the same compound appears at different temperatures in both train and test, which **inflates** performance.

Recommended split strategy:

1. **Train / validation / test** (e.g. 70/15/15 or 80/10/10; exact ratios matter less than split type).
2. **Development:** random split **by molecule** is fine for debugging the pipeline.
3. **Reporting “new molecule” performance:** use a **scaffold split** (e.g. Bemis–Murcko): scaffolds in test must not appear in train. This is usually **harder** and more realistic than random splitting for structure-based models. See, for example, discussions of scaffold splitting and generalization in molecular property prediction (e.g. [Practical Cheminformatics on splitting](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical-datasets.html), [Oxford Protein Informatics Group](https://www.blopig.com/blog/2021/06/out-of-distribution-generalisation-and-scaffold-splitting-in-molecular-property-prediction/)).

Use the **validation** set for hyperparameters and early stopping; reserve the **test** set for final numbers (or use nested cross-validation if publishing).

---

## 7. Training

- **Loss:** MSE or Huber on **`log(D)`** (Huber can help with outliers).
- **Metrics:** RMSE and MAE on **`log(D)`**; optionally RMSE/MAE on **linear \(D\)** after back-transform; \(R^2\) is common in papers—interpret alongside error metrics.
- **Regularization:** weight decay, dropout where appropriate; monitor **validation** loss, not training loss alone.

---

## 8. Benchmarking

Include **simple baselines** so improvements are meaningful:

- Global **mean** or **median** of `log(D)` on the training set.
- **Linear model:** fingerprints + temperature → `log(D)`.

Compare your test metrics to these baselines. Optionally compare qualitatively to **`Table S6`** (authors’ predictions vs experiment) as a **reference**, not as a substitute for your own held-out test protocol.

---

## 9. Error analysis

Before treating the model as production-ready:

- **Residuals vs temperature** — systematic bias at low or high \(T\) suggests poor extrapolation or missing **\(T\)** interaction.
- **Worst predictions** — outliers in the literature, unusual chemistry, or very large \(D\).
- **Scaffold vs random test gap** — a large gap is expected if scaffold split is stricter; it reflects real generalization to **new chemotypes**.

---

## 10. Inference and deployment

- **Inputs:** SMILES string, temperature in **kelvin**.
- **Pipeline:** canonicalize SMILES → encode structure → normalize **\(T\)** with **fixed training statistics** → model → `log(D)` → \(D\).
- **Validity:** state clearly that the model is trained for **water** only and for approximately the **observed temperature range**; predictions far outside that range are **extrapolation** and should be flagged or avoided.

---

## Summary checklist

1. Load **`Table S2`**; use **SMILES, Temperature, \(D\)**; confirm **units**.
2. Canonicalize SMILES; handle invalid rows; deduplicate **(SMILES, \(T\))**.
3. Predict **`log(D)`**; normalize **\(T\)** (train-only statistics).
4. Split **by molecule**; report **scaffold test** for “new compound” realism.
5. Baseline: **fingerprints + MLP**; iterate to GNN or pretrained encoder if needed.
6. Benchmark against trivial and linear models; analyze errors vs **\(T\)** and structure.
