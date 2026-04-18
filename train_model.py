import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from landmarks import normalize_landmarks


# load dataset
arthur = pd.read_csv("asl_landmarks_arthur.csv")
sara   = pd.read_csv("asl_landmarks_sara.csv")
davit  = pd.read_csv("asl_landmarks_davit.csv")

df = pd.concat([arthur, sara, davit], ignore_index=True)


assert df.isnull().sum().sum() == 0, "NaN values found in combined data"
assert df.shape[1] == 64, f"Expected 64 columns, got {df.shape[1]}"
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
print(f"Labels: {sorted(df['label'].unique())} ({df['label'].nunique()} unique)")
assert df['label'].nunique() == 24, f"Expected 24 labels, got {df['label'].nunique()}"


# BUILD FEATURES
coord_cols = [c for c in df.columns if c != "label"]
raw_coords = df[coord_cols].values.reshape(-1, 21, 3)

features = []
labels   = []
for i, pts in enumerate(raw_coords):
    vec = normalize_landmarks(pts)
    if vec is not None:
        features.append(vec)
        labels.append(df["label"].iloc[i])

X = np.array(features)
y = np.array(labels)
print(f"Feature matrix: {X.shape}  (expected ~{len(df)} x 84)")


# TRAIN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Augment training set with jittered copies to improve accuracy
AUGMENT_COPIES = 4      # each sample gets 4 noisy copies
JITTER_STD     = 0.015  # ~1.5% of the normalised scale

rng   = np.random.default_rng(42)
noise = rng.normal(0, JITTER_STD, (len(X_train) * AUGMENT_COPIES, X_train.shape[1]))
X_train_final = np.vstack([X_train, np.tile(X_train, (AUGMENT_COPIES, 1)) + noise])
y_train_final = np.concatenate([y_train, np.tile(y_train, AUGMENT_COPIES)])
print(f"Training set after augmentation: {X_train_final.shape}")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
rf.fit(X_train_final, y_train_final)


# EVALUATE Model
y_pred = rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

classes = sorted(rf.classes_)
cm = confusion_matrix(y_test, y_pred, labels=classes)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("ASL Random Forest — Confusion Matrix")
thresh = cm.max() / 2.0
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")



# SAVE model
joblib.dump(rf, "asl_model.pkl")
print("Saved asl_model.pkl")
