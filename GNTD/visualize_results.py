import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import warnings
import json

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================== 配置 ====================
figures_folder = "figures"
gene_compare_folder = os.path.join(figures_folder, "gene_comparison")
os.makedirs(gene_compare_folder, exist_ok=True)

# R环境设置
os.environ['R_HOME'] = '/home/wangluffy/miniconda3/envs/GNTD/lib/R'
import rpy2.robjects as robjects
robjects.r.library("mclust")
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
robjects.r['set.seed'](0)
Mclust = robjects.r['Mclust']

# 参数配置
l_num = [0, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
genes_to_visualize = ['LAMP2', 'GFAP', 'SYN1']

# ==================== 主分析循环 ====================
print("Loading results for each lambda...")
results = {}
mse_dict = {}
best_mse_list = []

for l in l_num:
    save_folder = f"GNTD_results_lambda_{l}"
    
    # 加载数据
    expr_mat = np.load(os.path.join(save_folder, "imputed_expr_mat.npy"))
    gene_names = np.load(os.path.join(save_folder, "gene_names.npy"), allow_pickle=True)
    x_coords = np.load(os.path.join(save_folder, "x_coords.npy"))
    y_coords = np.load(os.path.join(save_folder, "y_coords.npy"))
    
    # 加载原始表达矩阵（如果存在）
    raw_expr_mat = None
    raw_file = os.path.join(save_folder, "raw_expr_mat.npy")
    if os.path.exists(raw_file):
        raw_expr_mat = np.load(raw_file)
    
    # 读取MSE
    mse = np.nan
    info_file = os.path.join(save_folder, "training_info.txt")
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            for line in f:
                if 'Best MSE:' in line:
                    mse = float(line.split(':')[1].strip())
                    break
    if np.isnan(mse):
        mse_file = os.path.join(save_folder, "best_mse.npy")
        if os.path.exists(mse_file):
            mse = float(np.load(mse_file))
    if np.isnan(mse):
        json_file = os.path.join(save_folder, "results.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                mse = json.load(f).get('best_mse', np.nan)
    
    mse_dict[l] = mse
    best_mse_list.append(mse)
    
    # 数据清理
    expr_mat = np.nan_to_num(expr_mat, nan=0.0, posinf=0.0, neginf=0.0)
    expr_mat = np.clip(expr_mat, -1e10, 1e10)
    
    # PCA降维
    n_components = min(10, expr_mat.shape[1])
    expr_mat_hat = PCA(n_components=n_components).fit_transform(expr_mat)
    expr_mat_hat = np.nan_to_num(expr_mat_hat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Mclust自动聚类
    try:
        mclust = Mclust(rpy2.robjects.numpy2ri.numpy2rpy(expr_mat_hat), modelNames="EEE")
        clustering_labels = np.array(mclust[-2])
        n_clusters = len(np.unique(clustering_labels))
        
        if n_clusters >= 2:
            sil_score = silhouette_score(expr_mat_hat, clustering_labels)
            db_score = davies_bouldin_score(expr_mat_hat, clustering_labels)
            ch_score = calinski_harabasz_score(expr_mat_hat, clustering_labels)
        else:
            sil_score = db_score = ch_score = np.nan
    except Exception as e:
        print(f" Lambda {l} clustering failed: {e}")
        clustering_labels = None
        n_clusters = 0
        sil_score = db_score = ch_score = np.nan
    
    results[l] = {
        'expr_mat': expr_mat,
        'raw_expr_mat': raw_expr_mat,
        'gene_names': gene_names,
        'clustering_labels': clustering_labels,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'mse': mse,
        'n_clusters': n_clusters,
        'best_n_clusters': n_clusters,
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'calinski_harabasz': ch_score
    }
    
    print(f" Lambda {l}: MSE={mse:.6f}, Clusters={n_clusters}, Silhouette={sil_score:.4f}" 
          if not np.isnan(sil_score) else 
          f" Lambda {l}: MSE={mse:.6f}, Clusters={n_clusters}")

# ==================== 选择最佳lambda ====================
valid_mse = [i for i, m in enumerate(best_mse_list) if not np.isnan(m)]
best_idx = valid_mse[np.argmin([best_mse_list[i] for i in valid_mse])] if valid_mse else 0
best_lambda_mse = l_num[best_idx]

valid_sil = [(l, results[l]['silhouette']) for l in l_num if not np.isnan(results[l]['silhouette'])]
best_lambda_sil = max(valid_sil, key=lambda x: x[1])[0] if valid_sil else best_lambda_mse

print(f"\n最佳lambda (按MSE): {best_lambda_mse} (MSE={results[best_lambda_mse]['mse']:.6f})")
if valid_sil:
    print(f"最佳lambda (按Silhouette): {best_lambda_sil} (Silhouette={results[best_lambda_sil]['silhouette']:.4f})")

# ==================== 基因表达可视化 ====================
print("\n" + "="*60)
print("生成基因表达对比图...")
print("="*60)

best_result = results[best_lambda_mse]
gene_names_lower = np.char.lower(best_result['gene_names'])

for gene_name in genes_to_visualize:
    gene_idx = np.where(gene_names_lower == gene_name.lower())[0]
    if len(gene_idx) == 0:
        print(f" 警告: 基因 '{gene_name}' 未找到")
        continue
    gene_idx = gene_idx[0]
    actual_gene_name = best_result['gene_names'][gene_idx]
    
    imputed_expr = best_result['expr_mat'][:, gene_idx]
    raw_expr = None
    if best_result['raw_expr_mat'] is not None:
        raw_expr = (best_result['raw_expr_mat'][:, gene_idx] 
                   if best_result['raw_expr_mat'].ndim == 2 and best_result['raw_expr_mat'].shape[1] == len(gene_names_lower)
                   else None)
    
    # 单lambda对比图
    n_plots = 3 if raw_expr is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(15 if n_plots==3 else 12, 5))
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Raw
    if raw_expr is not None:
        ax = axes[0]
        raw_clean = np.nan_to_num(raw_expr, nan=0.0)
        vmin, vmax = np.percentile(raw_clean, [5, 95])
        sc = ax.scatter(best_result['x_coords'], best_result['y_coords'], c=raw_clean,
                        cmap='RdYlBu_r', s=20, alpha=0.7, vmin=vmin, vmax=vmax)
        ax.set_title(f'Raw: {actual_gene_name}')
        plt.colorbar(sc, ax=ax, label='Expression')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax2 = axes[1]
    else:
        ax2 = axes[0]
    
    # Imputed
    imp_clean = np.nan_to_num(imputed_expr, nan=0.0)
    vmin, vmax = np.percentile(imp_clean, [5, 95])
    sc = ax2.scatter(best_result['x_coords'], best_result['y_coords'], c=imp_clean,
                     cmap='RdYlBu_r', s=20, alpha=0.7, vmin=vmin, vmax=vmax)
    ax2.set_title(f'Imputed (λ={best_lambda_mse})')
    plt.colorbar(sc, ax=ax2, label='Expression')
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_ylim(ax2.get_ylim()[::-1])
    
    # Difference
    if raw_expr is not None:
        ax3 = axes[2]
        diff = imp_clean - np.nan_to_num(raw_expr, nan=0.0)
        vmin, vmax = np.percentile(diff, [5, 95])
        sc = ax3.scatter(best_result['x_coords'], best_result['y_coords'], c=diff,
                         cmap='RdBu_r', s=20, alpha=0.7, vmin=vmin, vmax=vmax)
        ax3.set_title('Difference (Imputed - Raw)')
        plt.colorbar(sc, ax=ax3, label='Difference')
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.set_ylim(ax3.get_ylim()[::-1])
    
    plt.suptitle(f'Gene Expression Comparison: {actual_gene_name}', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(gene_compare_folder, f'{actual_gene_name}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 已保存: {save_path}")

# 多lambda插补效果对比
print("\n生成多lambda插补效果对比图...")
for gene_name in genes_to_visualize:
    # 查找基因索引（取第一个找到的）
    gene_idx = None
    actual_gene_name = None
    for l in l_num:
        if results[l]['gene_names'] is not None:
            idxs = np.where(np.char.lower(results[l]['gene_names']) == gene_name.lower())[0]
            if len(idxs) > 0:
                gene_idx = idxs[0]
                actual_gene_name = results[l]['gene_names'][gene_idx]
                break
    if gene_idx is None:
        print(f" 基因 '{gene_name}' 未找到")
        continue
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    for i, l in enumerate(l_num):
        res = results[l]
        expr = res['expr_mat'][:, gene_idx]
        expr_clean = np.nan_to_num(expr, nan=0.0)
        vmin = np.percentile(expr_clean, 5) if expr_clean.std() > 0 else expr_clean.min()
        vmax = np.percentile(expr_clean, 95) if expr_clean.std() > 0 else expr_clean.max() + 1e-6
        
        sc = axes[i].scatter(res['x_coords'], res['y_coords'], c=expr_clean,
                             cmap='RdYlBu_r', s=10, alpha=0.7, vmin=vmin, vmax=vmax)
        axes[i].set_title(f'λ={l}\nMSE={res["mse"]:.5f}', fontsize=9)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        plt.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'{actual_gene_name} Imputation across Lambdas', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(gene_compare_folder, f'{actual_gene_name}_all_lambdas.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" 已保存: {save_path}")

# ==================== 保存总结与绘图 ====================
os.makedirs(figures_folder, exist_ok=True)

with open(os.path.join(figures_folder, "analysis_summary.txt"), 'w') as f:
    f.write("GNTD Spatial Transcriptomics Imputation Analysis Summary\n")
    f.write("="*60 + "\n\n")
    f.write("Evaluation Metrics:\n")
    f.write("- MSE: Lower is better\n")
    f.write("- Silhouette: Higher is better\n\n")
    for l in l_num:
        r = results[l]
        f.write(f"Lambda = {l}:\n")
        f.write(f"  MSE: {r['mse']:.6f}\n")
        f.write(f"  Clusters: {r['best_n_clusters']}\n")
        if not np.isnan(r['silhouette']):
            f.write(f"  Silhouette: {r['silhouette']:.4f}\n")
        f.write("\n")
    f.write(f"最佳lambda (MSE): {best_lambda_mse}\n")
    f.write(f"最佳lambda (Silhouette): {best_lambda_sil}\n")

# MSE vs Lambda
fig, ax = plt.subplots(figsize=(10, 6))
valid_l = [l for l, m in mse_dict.items() if not np.isnan(m)]
valid_mse = [mse_dict[l] for l in valid_l]
ax.plot(valid_l, valid_mse, 'o-', color='blue')
ax.set_xscale('log')
ax.set_xlabel('Lambda (log scale)')
ax.set_ylabel('Best MSE')
ax.set_title('MSE vs Lambda')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'mse_vs_lambda.png'), dpi=150)
plt.close()

# Silhouette vs Lambda
fig, ax = plt.subplots(figsize=(10, 6))
sil_scores = [results[l]['silhouette'] for l in l_num]
ax.plot(l_num, sil_scores, 'o-', color='green')
ax.set_xscale('log')
ax.set_xlabel('Lambda (log scale)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score vs Lambda')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'silhouette_vs_lambda.png'), dpi=150)
plt.close()

# 最佳MSE聚类图
best = results[best_lambda_mse]
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(best['x_coords'], best['y_coords'], c=best['clustering_labels'], 
                cmap='tab20', s=20, alpha=0.7)
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_title(f'Best Clustering by MSE (λ={best_lambda_mse})\n'
             f'MSE={best["mse"]:.6f}, Clusters={best["n_clusters"]}, Sil={best["silhouette"]:.4f}')
plt.colorbar(sc, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'clustering_best_mse.png'), dpi=150)
plt.close()

# 最佳Silhouette聚类图
if valid_sil:
    best_s = results[best_lambda_sil]
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(best_s['x_coords'], best_s['y_coords'], c=best_s['clustering_labels'], 
                    cmap='tab20', s=20, alpha=0.7)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title(f'Best Clustering by Silhouette (λ={best_lambda_sil})\n'
                 f'Sil={best_s["silhouette"]:.4f}, Clusters={best_s["n_clusters"]}')
    plt.colorbar(sc, ax=ax, label='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, 'clustering_best_silhouette.png'), dpi=150)
    plt.close()

# 所有lambda聚类对比
fig, axes = plt.subplots(2, 4, figsize=(16, 12))
axes = axes.flatten()
for i, l in enumerate(l_num):
    if results[l]['clustering_labels'] is not None:
        res = results[l]
        sc = axes[i].scatter(res['x_coords'], res['y_coords'], c=res['clustering_labels'],
                             cmap='tab20', s=10, alpha=0.7)
        axes[i].set_title(f'λ={l}\nClusters={res["best_n_clusters"]}\nSil={res["silhouette"]:.3f}')
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        plt.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04)
plt.suptitle('Clustering Results across All Lambda')
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'clustering_comparison_all.png'), dpi=150)
plt.close()

# 代表性lambda对比 (4个)
lambdas_to_show = [0, 0.01, 0.1, 1]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()
for i, l in enumerate(lambdas_to_show):
    if l in results and results[l]['clustering_labels'] is not None:
        res = results[l]
        sc = axes[i].scatter(res['x_coords'], res['y_coords'], c=res['clustering_labels'],
                             cmap='tab20', s=15, alpha=0.7)
        axes[i].set_title(f'λ={l}\nClusters={res["best_n_clusters"]}, Sil={res["silhouette"]:.3f}')
        axes[i].set_xticks([]); axes[i].set_yticks([])
        axes[i].set_ylim(axes[i].get_ylim()[::-1])
        plt.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04)
plt.suptitle('Clustering Comparison - Selected Lambdas')
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'clustering_comparison_selected.png'), dpi=150)
plt.close()

# ==================== 最终总结输出 ====================
print("\n" + "="*60)
print("分析完成！")
print("="*60)
print(f"最佳lambda (按MSE): {best_lambda_mse} (MSE={results[best_lambda_mse]['mse']:.6f})")
print(f"最佳lambda (按Silhouette): {best_lambda_sil} (Sil={results[best_lambda_sil]['silhouette']:.4f})")
print(f"\n结果保存位置: {figures_folder}/")
print(f"基因对比图保存位置: {gene_compare_folder}/")
print("="*60)