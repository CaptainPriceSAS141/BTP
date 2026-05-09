import matplotlib.pyplot as plt

def plot_curves(train_values, val_values, ylabel, save_path):

    plt.figure(figsize=(8, 5))

    plt.plot(train_values, label=f"Train {ylabel}")
    plt.plot(val_values, label=f"Val {ylabel}")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.legend()

    plt.savefig(save_path)

    plt.close()