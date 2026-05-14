import os
import time
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from models.vgg16 import get_vgg16
from models.resnet50 import get_resnet50

from utils.dataset import get_dataloaders
from utils.metrics import calculate_metrics
from utils.plotting import plot_curves

# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 32
EPOCHS = 50

LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# EXPERIMENTS
# =========================================================

EXPERIMENTS = [

    # Already completed
    # {
    #     "model": "vgg16",
    #     "dataset": "cifar10"
    # },

   # done

    {
        "model": "vgg16",
        "dataset": "cinic10"
    },

    {
        "model": "resnet50",
        "dataset": "cinic10"
    }
]

# =========================================================
# DIRECTORIES
# =========================================================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# =========================================================
# FINAL SUMMARY STORAGE
# =========================================================

overall_results = []

# =========================================================
# LOOP THROUGH ALL EXPERIMENTS
# =========================================================

for exp_num, exp in enumerate(EXPERIMENTS):

    MODEL_NAME = exp["model"]
    DATASET = exp["dataset"]

    print("\n" + "="*70)
    print(f"STARTING EXPERIMENT {exp_num+1}")
    print(f"MODEL   : {MODEL_NAME}")
    print(f"DATASET : {DATASET}")
    print("="*70)

    # =====================================================
    # DATA
    # =====================================================

    train_loader, test_loader = get_dataloaders(
        DATASET,
        BATCH_SIZE
    )

    # =====================================================
    # MODEL
    # =====================================================

    if MODEL_NAME == "vgg16":
        model = get_vgg16()

    elif MODEL_NAME == "resnet50":
        model = get_resnet50()

    else:
        raise ValueError("Invalid model")

    model = model.to(DEVICE)

    # =====================================================
    # LOSS + OPTIMIZER
    # =====================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # =====================================================
    # STORAGE
    # =====================================================

    history = []

    train_accs = []
    val_accs = []

    train_losses = []
    val_losses = []

    best_acc = 0

    total_training_start = time.time()

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        # =================================================
        # TRAIN
        # =================================================

        model.train()

        running_loss = 0

        correct = 0
        total = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"{MODEL_NAME}-{DATASET} Epoch {epoch+1}"
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)

        train_acc = correct / total

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        val_loss_running = 0

        correct = 0
        total = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss_running += loss.item()

                _, preds = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (preds == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss = val_loss_running / len(test_loader)

        val_acc = correct / total

        precision, recall, f1 = calculate_metrics(
            all_labels,
            all_preds
        )

        epoch_time = time.time() - epoch_start

        # =================================================
        # STORE
        # =================================================

        history.append({

            "epoch": epoch + 1,

            "train_loss": train_loss,
            "val_loss": val_loss,

            "train_acc": train_acc,
            "val_acc": val_acc,

            "precision": precision,
            "recall": recall,
            "f1_score": f1,

            "epoch_time_sec": epoch_time
        })

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # =================================================
        # SAVE BEST MODEL
        # =================================================

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                f"checkpoints/{MODEL_NAME}_{DATASET}.pth"
            )

        # =================================================
        # PRINT EPOCH RESULTS
        # =================================================

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1 Score   : {f1:.4f}")

        print(f"Epoch Time : {epoch_time:.2f} sec")

    # =====================================================
    # TOTAL TRAINING TIME
    # =====================================================

    total_training_time = time.time() - total_training_start

    # =====================================================
    # SAVE CSV
    # =====================================================

    df = pd.DataFrame(history)

    csv_path = f"results/metrics/{MODEL_NAME}_{DATASET}.csv"

    df.to_csv(csv_path, index=False)

    # =====================================================
    # SAVE PLOTS
    # =====================================================

    plot_curves(
        train_accs,
        val_accs,
        "Accuracy",
        f"results/plots/{MODEL_NAME}_{DATASET}_accuracy.png"
    )

    plot_curves(
        train_losses,
        val_losses,
        "Loss",
        f"results/plots/{MODEL_NAME}_{DATASET}_loss.png"
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    overall_results.append({

        "model": MODEL_NAME,
        "dataset": DATASET,

        "best_accuracy": best_acc,

        "precision": precision,
        "recall": recall,
        "f1_score": f1,

        "training_time_sec": total_training_time
    })

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED")
    print("="*70)

    print(f"MODEL           : {MODEL_NAME}")
    print(f"DATASET         : {DATASET}")

    print(f"BEST ACCURACY   : {best_acc:.4f}")
    print(f"PRECISION       : {precision:.4f}")
    print(f"RECALL          : {recall:.4f}")
    print(f"F1 SCORE        : {f1:.4f}")

    print(f"TRAINING TIME   : {total_training_time:.2f} sec")

    print(f"CSV SAVED       : {csv_path}")

    print("="*70)

# =========================================================
# FINAL SUMMARY
# =========================================================

summary_df = pd.DataFrame(overall_results)

summary_path = "results/metrics/final_summary.csv"

summary_df.to_csv(summary_path, index=False)

print("\n" + "#"*80)
print("ALL EXPERIMENTS FINISHED")
print("#"*80)

print(summary_df)

print(f"\nFinal summary saved to: {summary_path}")