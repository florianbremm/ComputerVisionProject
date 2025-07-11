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
    if len(set(labels_pred)) > 1:
        sil = silhouette_score(X, labels_pred)
        db = davies_bouldin_score(X, labels_pred)
    else:
        sil, db = np.nan, np.nan
    return sil, db

def evaluate_supervised(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return ari, nmi

def extract_and_reduce_embeddings(checkpoint_path, val_list, device, label_mode='dead_alive', num_frames_labels=10, batch_size=64, reducers=None):
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

    return reduced_results, labels

def plot_cluster_projections(reduced_results, cluster_labels, method_name="Unknown", save_dir=None):
    for reducer_name, X_2d in reduced_results.items():
        plt.figure(figsize=(7, 6))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, s=5, alpha=0.6)
        plt.title(f"{method_name} clustering shown via {reducer_name}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(Path(save_dir) / f"{method_name}_{reducer_name}_clusters.png", dpi=300)
        else:
            plt.show()

def run_clustering_evaluation(label_mode, num_frames, cluster_configs, device, model_path, val_list, reduced_results=None, batch_size=64):
    print(f"\n=== Running for label_mode='{label_mode}' | num_frames={num_frames} ===")
    
    # 1. Load model
    model = MoCoResNetBackbone()
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. Init dataset and labels
    dataset = CellDataset(video_list=val_list[:10], mode='inference', label_mode=label_mode, num_frames_labels=num_frames)
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
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(embeddings)

    # 5. Run clustering with different settings
    results = []
    for method_name, clusterer_factory in cluster_configs.items():
        print()
        model = clusterer_factory()
        labels_pred = model.fit_predict(X_pca)

        if reduced_results:
            plot_cluster_projections(reduced_results, labels_pred, method_name=method_name, save_dir="plots")

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
    checkpoint_path = Path("/scratch/cv-course-group-5/models/training4/model_epoch50.pth")  # adjust if needed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    _json_path = Path('train_test_split.json')

    # Load the file
    with open(_json_path, 'r') as f:
        _split_data = json.load(f)

    # Access the train and test entries
    train_list = _split_data.get("train", [])
    test_list = _split_data.get("test", [])
    val_list = _split_data.get("val", [])
    # Dimensionality reduction methods
    reducers = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, perplexity=30, random_state=42),
        "UMAP": umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        #  # --- t-SNE ---
        # "tSNE_perp10": TSNE(n_components=2, perplexity=10, n_iter=1500, learning_rate=300, init="pca", metric="euclidean", random_state=42),
        # "tSNE_perp30": TSNE(n_components=2, perplexity=30, n_iter=1500, learning_rate=300, init="pca", metric="euclidean", random_state=42),
        # "tSNE_perp50_cosine": TSNE(n_components=2, perplexity=50, n_iter=1500, learning_rate=300, init="pca", metric="cosine", random_state=42),

        # # --- UMAP ---
        # "UMAP_n15_d0.1": umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42),
        # "UMAP_n30_d0.0": umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric="euclidean", random_state=42),
        # "UMAP_n50_d0.3_cosine": umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.3, metric="cosine", random_state=42),
        # "UMAP_n100_d0.5_corr": umap.UMAP(n_components=2, n_neighbors=100, min_dist=0.5, metric="correlation", random_state=42),
    }
    # Define configs (you can expand this later)
    cluster_2 = {
        "HDBSCAN": lambda: HDBSCAN(min_cluster_size=100),
        "GMM-2": lambda: GaussianMixture(n_components=2, covariance_type='full', random_state=42),
        #"GMM": GaussianMixture(n_components=2, covariance_type='full', random_state=42, weights_init=[0.96, 0.04]),
        "Agglomerative-2": lambda: AgglomerativeClustering(n_clusters=2, linkage='complete'),    

        # "Agglomerative-Ward-2": AgglomerativeClustering(n_clusters=2, linkage="ward"),
        # "Agglomerative-Complete-2": AgglomerativeClustering(n_clusters=2, linkage="complete"),
        # "Agglomerative-Average-2": AgglomerativeClustering(n_clusters=2, linkage="average"),
        # "Agglomerative-Single-2": AgglomerativeClustering(n_clusters=2, linkage="single"),
    
    }

    cluster_3 = {
        "HDBSCAN": lambda: HDBSCAN(min_cluster_size=100),
        "GMM-3": lambda: GaussianMixture(n_components=3, covariance_type='full', random_state=42),
        "Agglomerative-3": lambda: AgglomerativeClustering(n_clusters=3, linkage='complete'),
        
        # "Agglomerative-Ward-3": AgglomerativeClustering(n_clusters=3, linkage="ward"),
        # "Agglomerative-Complete-3": AgglomerativeClustering(n_clusters=3, linkage="complete"),
        # "Agglomerative-Average-3": AgglomerativeClustering(n_clusters=3, linkage="average"),
        # "Agglomerative-Single-3": AgglomerativeClustering(n_clusters=3, linkage="single"),
    }
    reduced_results, labels = extract_and_reduce_embeddings(
        checkpoint_path=checkpoint_path,
        val_list=val_list[:10],
        device=device,
        label_mode='dead_alive',
        num_frames_labels=10,
        batch_size=64,
        reducers=reducers
    )
    # Run for both label modes
    results_all = []

    r = run_clustering_evaluation(
        label_mode='dead_alive',
        num_frames=10,
        cluster_configs=cluster_2,
        device=device,
        model_path=checkpoint_path,
        val_list=val_list[:10],
        reduced_results=reduced_results,
        batch_size=batch_size
    )
    results_all.extend(r)

    r = run_clustering_evaluation(
        label_mode='dead_alive_dividing',
        num_frames=10,
        cluster_configs=cluster_3,
        device=device,
        model_path=checkpoint_path,
        val_list=val_list[:10],
        reduced_results=reduced_results,
        batch_size=batch_size
    )
    results_all.extend(r)

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results_all)
    results_df.to_csv("clustering_eval_results.csv", index=False)

    print(results_df.to_string(index=False))