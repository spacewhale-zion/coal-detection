# Coal Detection from NIR Spectra

This project classifies near-infrared (NIR) spectra as coal (`1`) or non-coal (`0`). It is not a pixel-level hyperspectral image classifier. The dataset contains 10,496 spectra, 1,500 wavelength features per spectrum, and 24 material groups across 1000-2499 nm.

## Reproducibility

Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

The main data and model workflow is:

```powershell
python -m src.data.prepare_dataset
python -m src.splitting.dataset_split
python -m src.splitting.finalsplit
python -m src.models.train_final_model
python -m src.evaluation.evalute_final_model
```

All model selection experiments must group samples by `material`. Scalers and dimensionality-reduction steps are fitted only on training folds. The final holdout must not be used for tuning.

## Current Models

The primary stable model is:

```text
Raw spectrum -> StandardScaler -> PCA(2) -> RBF SVM
```

Additional evaluated models are implemented in `src/models/`:

- `pls_da.py`
- `random_forest_pca.py`
- `xgboost_pca.py`
- `cnn.py`

The repeated grouped validation currently favors SVM for stability, with XGBoost as a close secondary model. PLS-DA has high coal recall but is less stable across materials. The CNN and Random Forest are not currently competitive.

## Inference

The saved SVM artifacts can be used without refitting:

```powershell
python -m src.models.inference path\to\spectra.npy --output predictions.csv
```

The input must contain either one spectrum shaped `(1500,)` or multiple spectra shaped `(samples, 1500)`. The output contains the predicted label, class name, and SVM decision score.


## Evaluation Note

The original material-disjoint split contains 8,142 development samples and 2,354 holdout samples with no material overlap. During later model comparisons, the holdout was evaluated by multiple candidate models. Those stored holdout metrics are therefore comparison results, not a pristine final benchmark. A publication-quality final estimate requires a new external dataset or a newly locked holdout that is not used for model selection.
