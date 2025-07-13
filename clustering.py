# Clustering
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import pandas as pd

from CellDataset import CellDataset
from MoCoResNetBackbone import MoCoResNetBackbone

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

# os.chdir('ComputerVisionProject')

def evaluate_unsupervised(X, labels_pred):
    """
    :param X: Embeddings
    :param labels_pred: assigned cluster
    :return: silhouette_score, davies_bouldin_score
    """
    if len(set(labels_pred)) > 1:
        sil = silhouette_score(X, labels_pred)
        db = davies_bouldin_score(X, labels_pred)
    else:
        sil, db = np.nan, np.nan
    return sil, db

def evaluate_supervised(y_true, y_pred):
    """
    :param y_true: labels from ground truth mask
    :param y_pred: assigned cluster
    :return: adjusted_rand_score, normalized_mutual_info_score
    """
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return ari, nmi

def extract_and_reduce_embeddings(checkpoint_path, val_list, device, label_mode='dead_alive', num_frames_labels=10, batch_size=64, reducers=None):
    """
    embedd all cell-images from a list of videos, reduce their dimensionality and plot them with their ground truth
    :param checkpoint_path: the model to use for embedding
    :param val_list: the video list
    :param device: cuda or cpu
    :param label_mode: "dead_alive" / "dead_alive_dividing" / "frames_till_death" whether to differentiate between alive and dividing for the label ("dead_alive_dividing") or to label a cell as dead before its actual death ("frames_till_death")
    :param num_frames_labels: how many frames before it's annotated dead a cell should be labeled as dead
    :param batch_size: batch size for inference
    :param reducers: dict[str, dimensionality reduction methods to use]
    :return:
    """
    # 1. Load model
    model = MoCoResNetBackbone()
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. Load dataset and DataLoader
    dataset = CellDataset(video_list=val_list, mode='inference', label_mode=label_mode, num_frames_labels=num_frames_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 3. Extract embeddings and labels
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Extracting embeddings"):
            imgs = imgs.to(device, non_blocking=True)
            embeddings = model.encode_query(imgs).cpu().numpy()
            labels = labels.cpu().numpy()

            all_embeddings.append(embeddings)
            all_labels.append(labels)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 4. Run dimensionality reduction
    reduced_results = {}
    if reducers is not None:
        for name, reducer in reducers.items():
            print(f"Running {name}...")
            reduced = reducer.fit_transform(embeddings)
            reduced_results[name] = reduced

    method_name_label = 'GT-Labels-' + label_mode
    plot_cluster_projections(reduced_results=reduced_results, cluster_labels=labels, method_name=method_name_label, save_dir='plots', gt=True)

    return reduced_results, labels

def plot_cluster_projections(reduced_results, cluster_labels, method_name="Unknown", save_dir=None, gt=False):
    """
    plots cluster for different dimensionality reduction methods
    :param reduced_results: embeddings after dimensionality reduction
    :param cluster_labels: assigned cluster
    :param method_name: Name of method
    :param save_dir: where to save the figure
    :return:
    """
    for reducer_name, X_2d in reduced_results.items():
        plt.figure(figsize=(7, 6))
        unique_labels = np.unique(cluster_labels)

        if gt:
            unique_labels = unique_labels[::-1]

        for label in unique_labels:
            alpha = 0.3
            if gt and label == 0:
                alpha = 0.8
            mask = cluster_labels == label
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                s=4,
                alpha=alpha,
                label=f"{label}" if label != -1 else "Noise"
            )

        plt.legend(title="Cluster Label", loc='upper right', markerscale=3)   
        # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='viridis', s=5, alpha=0.6)
        plt.title(f"{method_name} clustering shown via {reducer_name}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(Path(save_dir) / f"{method_name}_{reducer_name}_clusters.png", dpi=300)
            plt.close()
        else:
            plt.show()
            plt.close()

def run_clustering_evaluation(label_mode, num_frames, cluster_configs, device, model_path, val_list, reduced_results=None, batch_size=64, pca_dimension=10):
    """
    Run a clustering
    :param label_mode: "dead_alive" / "dead_alive_dividing" / "frames_till_death" whether to differentiate between alive and dividing for the label ("dead_alive_dividing") or to label a cell as dead before its actual death ("frames_till_death")
    :param num_frames_labels: how many frames before it's annotated dead a cell should be labeled as dead
    :param cluster_configs: dict [str, lamda -> Clustering Methode]
    :param device: cuda or cpu
    :param model_path: where to save the model
    :param val_list: the video list
    :param reduced_results: embeddings after dimensionality reduction
    :param batch_size: batch size for inference
    :param pca_dimension: dimensionality of PCA
    :return:
    """
    print(f"\n=== Running for label_mode='{label_mode}' | num_frames={num_frames} ===")
    
    # 1. Load model
    model = MoCoResNetBackbone()
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. Init dataset and labels
    dataset = CellDataset(video_list=val_list, mode='inference', label_mode=label_mode, num_frames_labels=num_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 3. Extract embeddings
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Extracting embeddings for '{label_mode}'"):
            imgs = imgs.to(device, non_blocking=True)
            embeddings = model.encode_query(imgs).cpu().numpy()
            labels = labels.cpu().numpy()
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 4. Apply PCA
    pca = PCA(n_components=pca_dimension)
    X_pca = pca.fit_transform(embeddings)

    # 5. Run clustering with different settings
    results = []
    for method_name, clusterer_factory in cluster_configs.items():
        print("")
        model = clusterer_factory()
        labels_pred = model.fit_predict(X_pca)
        print('trained')

        if reduced_results:
            method_name_pca = method_name+'_pca_'+str(pca_dimension)
            plot_cluster_projections(reduced_results, labels_pred, method_name=method_name_pca, save_dir="plots")

        sil, db = evaluate_unsupervised(X_pca, labels_pred)
        ari, nmi = evaluate_supervised(labels, labels_pred)
        n_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)

        results.append({
            "LabelMode": label_mode,
            "Method": method_name,
            "#Clusters": n_clusters,
            "Silhouette": sil,
            "DBIndex": db,
            "ARI": ari,
            "NMI": nmi
        })

    return results

if __name__ == '__main__':
    # ========== Configuration ==========
    checkpoint_path = Path("/scratch/cv-course-group-5/models/training5/model_epoch50.pth")  # adjust if needed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    _json_path = Path('video_lists/train_test_split.json')

    # Load the file
    with open(_json_path, 'r') as f:
        _split_data = json.load(f)

    # Access the train and test entries
    train_list = _split_data.get("train", [])
    test_list = _split_data.get("test", [])
    val_list = _split_data.get("val", [])
    # Dimensionality reduction methods
    reducers = {
        # === PCA (Linear Baseline) ===
        "PCA_2D": PCA(n_components=2),

        # === t-SNE with Cosine Distance ===
        "tSNE_perp30_cos": TSNE(n_components=2, perplexity=30, max_iter=1500,
                                learning_rate=300, init="pca", metric="cosine", random_state=42),
        "tSNE_perp50_cos": TSNE(n_components=2, perplexity=50, max_iter=1500,
                                learning_rate=300, init="pca", metric="cosine", random_state=42),

        # === Cosine Metric ===
        "UMAP_30_0.0_cos": umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, metric="cosine", random_state=42),
        "UMAP_50_0.3_cos": umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.4, metric="cosine", random_state=42),

        # === Correlation Metric ===
        "UMAP_30_0.1_corr": umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.2, metric="correlation", random_state=42),
    }
    # Define configs
    cluster_2 = {
        "HDBSCAN": lambda: HDBSCAN(min_cluster_size=70, metric='cosine'),
        "GMM-2": lambda: GaussianMixture(n_components=2, covariance_type='full', random_state=42),
        "Agglomerative-Complete-2": lambda: AgglomerativeClustering(n_clusters=2, linkage="complete", metric='cosine'),
    }

    cluster_3 = {
        "HDBSCAN": lambda: HDBSCAN(min_cluster_size=70, metric='cosine'),
        "GMM-3": lambda: GaussianMixture(n_components=3, covariance_type='full', random_state=42),
        "Agglomerative-Complete-3": lambda: AgglomerativeClustering(n_clusters=3, linkage="complete", metric='cosine'),
    }

    val_list_end = 9

    reduced_results, labels = extract_and_reduce_embeddings(
        checkpoint_path=checkpoint_path,
        val_list=val_list[:val_list_end],
        device=device,
        label_mode='dead_alive',
        num_frames_labels=0,
        batch_size=64,
        reducers=reducers
    )
    # Run for both label modes
    results_all = []

    r = run_clustering_evaluation(
        label_mode='dead_alive',
        num_frames=0,
        cluster_configs=cluster_2,
        device=device,
        model_path=checkpoint_path,
        val_list=val_list[:val_list_end],
        reduced_results=reduced_results,
        batch_size=batch_size
    )
    results_all.extend(r)

    r = run_clustering_evaluation(
        label_mode='dead_alive_dividing',
        num_frames=0,
        cluster_configs=cluster_3,
        device=device,
        model_path=checkpoint_path,
        val_list=val_list[:val_list_end],
        reduced_results=reduced_results,
        batch_size=batch_size
    )
    results_all.extend(r)

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results_all)
    results_df.to_csv("clustering_eval_results.csv", index=False)

    print(results_df.to_string(index=False))