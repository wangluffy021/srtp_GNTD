# GNTD_visualization_clustering_debug.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set R environment for mclust
os.environ['R_HOME'] = '/home/wangluffy/miniconda3/envs/GNTD/lib/R'
import rpy2.robjects as robjects
robjects.r.library("mclust")
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
r_random_seed = robjects.r['set.seed']
r_random_seed(0)
Mclust = robjects.r['Mclust']

def visualize_and_cluster(save_folder="GNTD_results_lambda_0.01"):
    """
    Load imputed results and perform clustering and visualization
    """
    
    # Load saved data
    print(f"Loading results from {save_folder}/")
    expr_mat = np.load(os.path.join(save_folder, "imputed_expr_mat.npy"))
    gene_names = np.load(os.path.join(save_folder, "gene_names.npy"), allow_pickle=True)
    x_coords = np.load(os.path.join(save_folder, "x_coords.npy"))
    y_coords = np.load(os.path.join(save_folder, "y_coords.npy"))
    mapping = np.load(os.path.join(save_folder, "mapping.npy"), allow_pickle=True)
    
    # Load training info
    with open(os.path.join(save_folder, "training_info.txt"), 'r') as f:
        print(f.read())
    
    # ========== DEBUG: Check ground truth labels ==========
    print("\n" + "="*50)
    print("DEBUGGING INFORMATION")
    print("="*50)
    spot_idx = np.where(mapping[:, -1] != -2)[0]
    ground_truth = mapping[spot_idx, -1]
    
    print(f"Number of spots with ground truth: {len(spot_idx)}")
    print(f"Unique ground truth labels: {np.unique(ground_truth)}")
    print(f"Ground truth distribution: {np.bincount(ground_truth.astype(int))}")
    print(f"Number of clusters in ground truth: {len(np.unique(ground_truth))}")
    
    # PCA
    print("\nStarting PCA...")
    n_components = min(10, expr_mat.shape[1])
    expr_mat_hat = PCA(n_components=n_components).fit_transform(expr_mat)
    print(f"PCA completed. Shape after PCA: {expr_mat_hat.shape}")
    
    # Mclust clustering
    print("Starting Mclust clustering...")
    # Try different model names for better clustering
    model_names = ["EEE", "VVV", "EEI", "VEI", "EII"]
    best_mclust = None
    best_bic = -np.inf
    
    for model in model_names:
        try:
            mclust = Mclust(rpy2.robjects.numpy2ri.numpy2rpy(expr_mat_hat), 15, model)
            bic = mclust[-4][0]  # BIC value
            if bic > best_bic:
                best_bic = bic
                best_mclust = mclust
                print(f"  Model {model}: BIC={bic:.2f}, n_clusters={len(np.unique(mclust[-2]))}")
        except Exception as e:
            print(f"  Model {model}: Failed - {str(e)}")
    
    if best_mclust is None:
        print("Using default model EEE")
        best_mclust = Mclust(rpy2.robjects.numpy2ri.numpy2rpy(expr_mat_hat), 15, "EEE")
    
    clustering_labels = best_mclust[-2]
    print(f"Clustering completed. Number of clusters found: {len(np.unique(clustering_labels))}")
    print(f"Cluster sizes: {np.bincount(clustering_labels)}")
    
    # Calculate ARI
    ari_score = adjusted_rand_score(ground_truth, clustering_labels)
    print(f"\nAdjusted Rand Index (ARI): {ari_score:.4f}")
    
    # Check if ARI is zero due to label alignment issues
    if ari_score == 0:
        print("\n⚠️  WARNING: ARI is 0. Possible reasons:")
        print("  1. Ground truth has only one cluster (all spots same type)")
        print("  2. Clustering found only one cluster")
        print("  3. Complete disagreement between clustering and ground truth")
        
        # Try to see if there's any agreement
        from sklearn.metrics import rand_score, mutual_info_score
        rand = rand_score(ground_truth, clustering_labels)
        mi = mutual_info_score(ground_truth, clustering_labels)
        print(f"  Rand Index: {rand:.4f}")
        print(f"  Mutual Information: {mi:.4f}")
    
    # Create visualization folder
    viz_folder = os.path.join(save_folder, "visualizations")
    os.makedirs(viz_folder, exist_ok=True)
    
    # Visualization 1: Spatial clustering
    print("\nCreating spatial clustering plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground truth
    scatter1 = axes[0].scatter(x_coords[spot_idx], y_coords[spot_idx], 
                               c=ground_truth, cmap='tab20', s=10)
    axes[0].set_ylim(axes[0].get_ylim()[::-1])
    axes[0].tick_params('both', left=False, labelleft=False, bottom=False, labelbottom=False)
    axes[0].set_title(f'Ground Truth ({len(np.unique(ground_truth))} clusters)', fontsize=12)
    
    # Clustering result
    scatter2 = axes[1].scatter(x_coords, y_coords, c=clustering_labels, cmap='tab20', s=10)
    axes[1].set_ylim(axes[1].get_ylim()[::-1])
    axes[1].tick_params('both', left=False, labelleft=False, bottom=False, labelbottom=False)
    axes[1].set_title(f'Mclust Clustering ({len(np.unique(clustering_labels))} clusters)\nARI={ari_score:.3f}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, "spatial_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {viz_folder}/spatial_comparison.png")
    
    # Visualization 2: PCA scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(expr_mat_hat[:, 0], expr_mat_hat[:, 1], 
                        c=clustering_labels, cmap='tab20', s=10, alpha=0.6)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'PCA visualization of clusters (ARI={ari_score:.3f})', fontsize=14)
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, "pca_clustering.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {viz_folder}/pca_clustering.png")
    
    # Save clustering labels and info
    np.save(os.path.join(viz_folder, "clustering_labels.npy"), clustering_labels)
    np.save(os.path.join(viz_folder, "ground_truth_labels.npy"), ground_truth)
    
    with open(os.path.join(viz_folder, "clustering_info.txt"), 'w') as f:
        f.write(f"ARI Score: {ari_score:.6f}\n")
        f.write(f"Number of ground truth clusters: {len(np.unique(ground_truth))}\n")
        f.write(f"Number of detected clusters: {len(np.unique(clustering_labels))}\n")
        f.write(f"Ground truth distribution: {np.bincount(ground_truth.astype(int))}\n")
        f.write(f"Detected cluster sizes: {np.bincount(clustering_labels)}\n")
    
    print(f"\nAll results saved to {viz_folder}/")
    return clustering_labels, ari_score, ground_truth

def analyze_expression_patterns(save_folder="GNTD_results_lambda_0.01"):
    """
    Analyze expression patterns without relying on ground truth
    """
    print("\n" + "="*50)
    print("Analyzing expression patterns")
    print("="*50)
    
    # Load data
    expr_mat = np.load(os.path.join(save_folder, "imputed_expr_mat.npy"))
    gene_names = np.load(os.path.join(save_folder, "gene_names.npy"), allow_pickle=True)
    x_coords = np.load(os.path.join(save_folder, "x_coords.npy"))
    y_coords = np.load(os.path.join(save_folder, "y_coords.npy"))
    
    # Simple k-means clustering for comparison
    from sklearn.cluster import KMeans
    
    print("Performing K-means clustering as alternative...")
    for n_clusters in [3, 5, 7, 10]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans_labels = kmeans.fit_predict(expr_mat)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(x_coords, y_coords, c=kmeans_labels, cmap='tab20', s=10)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.tick_params('both', left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.set_title(f'K-means Clustering (k={n_clusters})', fontsize=14)
        plt.colorbar(scatter, ax=ax)
        plt.tight_layout()
        
        viz_folder = os.path.join(save_folder, "visualizations")
        os.makedirs(viz_folder, exist_ok=True)
        plt.savefig(os.path.join(viz_folder, f"kmeans_clusters_{n_clusters}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  k={n_clusters}: Saved to {viz_folder}/kmeans_clusters_{n_clusters}.png")
    
    # Find highly variable genes and visualize
    print("\nVisualizing top variable genes...")
    gene_vars = np.var(expr_mat, axis=0)
    top_genes_idx = np.argsort(gene_vars)[-6:]  # Top 6 most variable genes
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, gene_idx in enumerate(top_genes_idx):
        ax = axes[i]
        scatter = ax.scatter(x_coords, y_coords, c=expr_mat[:, gene_idx], 
                            cmap='RdYlBu_r', s=10)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.tick_params('both', left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.set_title(f'{gene_names[gene_idx]}', fontsize=12)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, "top_variable_genes.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {viz_folder}/top_variable_genes.png")

if __name__ == "__main__":
    # Run debug version
    lambda_to_visualize = 0.01
    clustering_labels, ari, ground_truth = visualize_and_cluster(f"GNTD_results_lambda_{lambda_to_visualize}")
    
    # Additional analysis
    analyze_expression_patterns(f"GNTD_results_lambda_{lambda_to_visualize}")
    
    print("\n" + "="*50)
    print("Analysis complete! Check the visualizations folder for results.")
    print("="*50)