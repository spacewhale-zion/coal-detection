import json
import joblib
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "final_split"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

RANDOM_STATE = 42
N_SPLITS = 5
EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpectralCNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5),
			nn.BatchNorm1d(16),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(32, 64, kernel_size=5, padding=2),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(1)
		)
		self.classifier = nn.Linear(64, 1)

	def forward(self, inputs):
		features = self.features(inputs).squeeze(-1)
		return self.classifier(features).squeeze(-1)


def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_data():
	X = np.load(FINAL_SPLIT_DIR / "X_development.npy")
	y = np.load(FINAL_SPLIT_DIR / "y_development.npy")
	X_test = np.load(FINAL_SPLIT_DIR / "X_final_test.npy")
	y_test = np.load(FINAL_SPLIT_DIR / "y_final_test.npy")
	development_metadata = pd.read_csv(
		FINAL_SPLIT_DIR / "metadata_development.csv"
	)
	test_metadata = pd.read_csv(
		FINAL_SPLIT_DIR / "metadata_final_test.csv"
	)
	return X, y, X_test, y_test, development_metadata, test_metadata


def make_loader(X, y, shuffle):
	tensor_X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
	tensor_y = torch.from_numpy(y.astype(np.float32))
	return DataLoader(
		TensorDataset(tensor_X, tensor_y),
		batch_size=BATCH_SIZE,
		shuffle=shuffle,
		num_workers=0
	)


def train_model(X_train, y_train, X_validation, y_validation):
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
	X_validation_scaled = scaler.transform(X_validation).astype(np.float32)

	model = SpectralCNN().to(DEVICE)
	positive = max(np.sum(y_train == 1), 1)
	negative = max(np.sum(y_train == 0), 1)
	loss_function = nn.BCEWithLogitsLoss(
		pos_weight=torch.tensor(
			[negative / positive],
			dtype=torch.float32,
			device=DEVICE
		)
	)
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=LEARNING_RATE,
		weight_decay=1e-4
	)
	train_loader = make_loader(X_train_scaled, y_train, shuffle=True)
	validation_loader = make_loader(
		X_validation_scaled,
		y_validation,
		shuffle=False
	)

	best_state = None
	best_loss = float("inf")
	best_epoch = 1
	epochs_without_improvement = 0

	for epoch in range(1, EPOCHS + 1):
		model.train()
		for batch_X, batch_y in train_loader:
			batch_X = batch_X.to(DEVICE)
			batch_y = batch_y.to(DEVICE)
			optimizer.zero_grad()
			loss = loss_function(model(batch_X), batch_y)
			loss.backward()
			optimizer.step()

		validation_loss = evaluate_loss(
			model,
			validation_loader,
			loss_function
		)
		if validation_loss < best_loss:
			best_loss = validation_loss
			best_state = {
				key: value.detach().cpu().clone()
				for key, value in model.state_dict().items()
			}
			best_epoch = epoch
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1

		if epochs_without_improvement >= PATIENCE:
			break

	model.load_state_dict(best_state)
	return model, scaler, best_epoch


def evaluate_loss(model, loader, loss_function):
	model.eval()
	total_loss = 0.0
	total_samples = 0
	with torch.no_grad():
		for batch_X, batch_y in loader:
			batch_X = batch_X.to(DEVICE)
			batch_y = batch_y.to(DEVICE)
			total_loss += float(loss_function(
				model(batch_X), batch_y
			)) * len(batch_y)
			total_samples += len(batch_y)
	return total_loss / max(total_samples, 1)


def predict(model, scaler, X):
	X_scaled = scaler.transform(X).astype(np.float32)
	loader = make_loader(
		X_scaled,
		np.zeros(len(X), dtype=np.float32),
		shuffle=False
	)
	scores = []
	model.eval()
	with torch.no_grad():
		for batch_X, _ in loader:
			scores.extend(
				torch.sigmoid(model(batch_X.to(DEVICE)))
				.cpu()
				.numpy()
			)
	scores = np.asarray(scores)
	return scores, (scores >= 0.5).astype(np.int64)


def calculate_metrics(y_true, y_pred):
	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(
			y_true, y_pred, zero_division=0
		),
		"recall": recall_score(
			y_true, y_pred, zero_division=0
		),
		"f1": f1_score(y_true, y_pred, zero_division=0)
	}


def train_final_model(X, y, epochs):
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X).astype(np.float32)
	model = SpectralCNN().to(DEVICE)
	positive = max(np.sum(y == 1), 1)
	negative = max(np.sum(y == 0), 1)
	loss_function = nn.BCEWithLogitsLoss(
		pos_weight=torch.tensor(
			[negative / positive],
			dtype=torch.float32,
			device=DEVICE
		)
	)
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=LEARNING_RATE,
		weight_decay=1e-4
	)
	loader = make_loader(X_scaled, y, shuffle=True)

	model.train()
	for _ in range(epochs):
		for batch_X, batch_y in loader:
			batch_X = batch_X.to(DEVICE)
			batch_y = batch_y.to(DEVICE)
			optimizer.zero_grad()
			loss = loss_function(model(batch_X), batch_y)
			loss.backward()
			optimizer.step()
	return model, scaler


def main():
	set_seed(RANDOM_STATE)
	X, y, X_test, y_test, development_metadata, test_metadata = load_data()
	groups = development_metadata["material"].to_numpy()
	cv = StratifiedGroupKFold(
		n_splits=N_SPLITS,
		shuffle=True,
		random_state=RANDOM_STATE
	)
	fold_rows = []
	selected_epochs = []

	print("=" * 70)
	print("1D CNN — GROUPED CROSS-VALIDATION")
	print("=" * 70)
	print(f"Device: {DEVICE}")
	print(f"Development samples: {len(X)}")

	for fold, (train_idx, validation_idx) in enumerate(
		cv.split(X, y, groups),
		start=1
	):
		set_seed(RANDOM_STATE + fold)
		model, scaler, best_epoch = train_model(
			X[train_idx],
			y[train_idx],
			X[validation_idx],
			y[validation_idx]
		)
		_, predictions = predict(
			model,
			scaler,
			X[validation_idx]
		)
		fold_metrics = calculate_metrics(
			y[validation_idx],
			predictions
		)
		fold_metrics.update({
			"fold": fold,
			"material": groups[validation_idx][0],
			"best_epoch": best_epoch
		})
		fold_rows.append(fold_metrics)
		selected_epochs.append(best_epoch)
		print(
			f"Fold {fold}: material={groups[validation_idx][0]}, "
			f"accuracy={fold_metrics['accuracy']:.4f}, "
			f"f1={fold_metrics['f1']:.4f}, "
			f"epoch={best_epoch}"
		)

	fold_results = pd.DataFrame(fold_rows)
	final_epochs = max(1, int(round(np.median(selected_epochs))))
	final_model, final_scaler = train_final_model(X, y, final_epochs)
	_, final_predictions = predict(final_model, final_scaler, X_test)
	final_metrics = calculate_metrics(y_test, final_predictions)
	confusion = confusion_matrix(y_test, final_predictions)
	tn, fp, fn, tp = confusion.ravel()
	final_metrics.update({
		"epochs": final_epochs,
		"training_samples": int(len(X)),
		"test_samples": int(len(X_test)),
		"confusion_matrix": {
			"true_negative": int(tn),
			"false_positive": int(fp),
			"false_negative": int(fn),
			"true_positive": int(tp)
		}
	})

	MODEL_DIR.mkdir(parents=True, exist_ok=True)
	METRICS_DIR.mkdir(parents=True, exist_ok=True)
	PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
	fold_results.to_csv(
		METRICS_DIR / "cnn_grouped_cv_results.csv",
		index=False
	)
	with open(METRICS_DIR / "cnn_test_metrics.json", "w") as file:
		json.dump(final_metrics, file, indent=4)
	joblib.dump(final_scaler, MODEL_DIR / "cnn_scaler.pkl")
	torch.save({
		"model_state_dict": final_model.state_dict(),
		"epochs": final_epochs,
		"input_features": int(X.shape[1])
	}, MODEL_DIR / "cnn_model.pt")

	test_predictions = test_metadata.copy()
	test_predictions["true_label"] = y_test
	test_predictions["predicted_label"] = final_predictions
	test_predictions["correct"] = (
		test_predictions["true_label"]
		== test_predictions["predicted_label"]
	)
	test_predictions.to_csv(
		PREDICTIONS_DIR / "cnn_predictions.csv",
		index=False
	)

	print("\nCNN CV summary")
	print(fold_results[["accuracy", "precision", "recall", "f1"]].mean())
	print(f"\nFinal test results\n{json.dumps(final_metrics, indent=4)}")
	print("\nClassification report")
	print(classification_report(
		y_test,
		final_predictions,
		target_names=["Non-Coal", "Coal"],
		zero_division=0
	))
	print("CNN artifacts saved.")


if __name__ == "__main__":
	main()
