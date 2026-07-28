import os
import matplotlib
matplotlib.use("Agg")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from train import MusicLSTM, MusicDataset


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _class_labels():
    return [str(i + 60) for i in range(36)]


def _save_line_plot(target_mean, pred_mean, charts_dir):
    labels = np.arange(36) + 60
    plt.figure(figsize=(14, 5))
    plt.plot(labels, target_mean, marker="o", linewidth=2, label="Target")
    plt.plot(labels, pred_mean, marker="o", linewidth=2, label="Prediction")
    plt.xlabel("MIDI Pitch")
    plt.ylabel("Normalized Frequency")
    plt.title("Average Target vs Prediction Distribution")
    plt.xticks(labels[::2], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(charts_dir, "average_target_vs_prediction.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def _heatmap_height(num_rows):
    return min(18, max(4, 0.12 * num_rows))


def _save_heatmap(data, title, filename, charts_dir):
    plt.figure(figsize=(16, _heatmap_height(len(data))))
    sns.heatmap(
        data,
        cmap="viridis",
        cbar_kws={"label": "Normalized Frequency"},
        xticklabels=_class_labels(),
        yticklabels=False,
    )
    plt.xlabel("MIDI Pitch")
    plt.ylabel("Batch")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(charts_dir, filename),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def _save_centroid_heatmap(centers, charts_dir):
    plt.figure(figsize=(16, max(4, 0.8 * len(centers))))
    sns.heatmap(
        centers,
        cmap="magma",
        cbar_kws={"label": "Centroid Value"},
        xticklabels=_class_labels(),
        yticklabels=[f"Cluster {i}" for i in range(len(centers))],
    )
    plt.xlabel("MIDI Pitch")
    plt.ylabel("Cluster")
    plt.title("KMeans Cluster Centroids")
    plt.tight_layout()
    plt.savefig(
        os.path.join(charts_dir, "kmeans_centroids_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def _save_distance_plot(distances, charts_dir):
    plt.figure(figsize=(14, 5))
    plt.plot(np.arange(len(distances)), distances, linewidth=1.5)
    plt.xlabel("Batch")
    plt.ylabel("Distance")
    plt.title("Nearest Cluster Distance per Batch")
    plt.tight_layout()
    plt.savefig(
        os.path.join(charts_dir, "batch_nearest_cluster_distances.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def evaluate_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "dataframes", "final_features.pkl")
    model_path = os.path.join(base_dir, "models", "music_lstm.pth")
    charts_dir = os.path.join(base_dir, "evaluation_charts")
    _ensure_dir(charts_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_pickle(data_path)
    dataset = MusicDataset(df, seq_length=50)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    input_dim = 17
    vocab_size = 36
    num_layers = 3
    hidden_dim = 128

    model = MusicLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_distributions = []
    predicted_distributions = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            predicted = torch.argmax(outputs, dim=1)

            target_hist = np.bincount(
                batch_y.cpu().numpy(),
                minlength=36,
            ) / len(batch_y)

            pred_hist = np.bincount(
                predicted.cpu().numpy(),
                minlength=36,
            ) / len(predicted)

            target_distributions.append(target_hist)
            predicted_distributions.append(pred_hist)

    if not target_distributions:
        raise ValueError("No evaluation batches were generated from the dataset.")

    target_distributions = np.array(target_distributions)
    predicted_distributions = np.array(predicted_distributions)

    n_clusters = min(5, len(target_distributions))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(target_distributions)

    distances = kmeans.transform(predicted_distributions)
    min_distances = np.min(distances, axis=1)
    mean_distance = np.mean(min_distances)

    target_mean = target_distributions.mean(axis=0)
    pred_mean = predicted_distributions.mean(axis=0)

    _save_line_plot(target_mean, pred_mean, charts_dir)
    _save_heatmap(
        target_distributions,
        "Target Distribution Heatmap by Batch",
        "target_distribution_heatmap.png",
        charts_dir,
    )
    _save_heatmap(
        predicted_distributions,
        "Prediction Distribution Heatmap by Batch",
        "prediction_distribution_heatmap.png",
        charts_dir,
    )
    _save_centroid_heatmap(kmeans.cluster_centers_, charts_dir)
    _save_distance_plot(min_distances, charts_dir)

    print(f"Cluster Evaluation Mean Distance: {mean_distance:.4f}")
    print(f"Charts saved to: {charts_dir}")


if __name__ == "__main__":
    evaluate_model()
