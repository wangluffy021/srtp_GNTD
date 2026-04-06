import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import warnings
import json

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 创建figures文件夹
figures_folder = "figures"
os.makedirs(figures_folder, exist_ok=True)

# 创建gene_comparison子文件夹
gene_compare_folder = os.path.join(figures_folder, "gene_comparison")
os.makedirs(gene_compare_folder, exist_ok=True)

# 设置R环境
os.environ['R_HOME'] = '/home/wangluffy/miniconda3/envs/GNTD/lib/R'
import rpy2.robjects as robjects
robjects.r.library("mclust")
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
r_random_seed = robjects.r['set.seed']
r_random_seed(0)
Mclust = robjects.r['Mclust']

# Lambda值列表
l_num = [0, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]

# 指定要可视化的基因列表
genes_to_visualize = ['LAMP2', 'CD68', 'GFAP', 'SYN1']  # 可根据需要修改

# 存储结果
results = {}
best_mse_list = []
mse_dict = {}

print("Loading results for each lambda...")
for l in l_num:
    save_folder = f"GNTD_results_lambda_{l}"
    
    # 加载数据
    expr_mat = np.load(os.path.join(save_folder, "imputed_expr_mat.npy"))
    gene_names = np.load(os.path.join(save_folder, "gene_names.npy"), allow_pickle=True)
    x_coords = np.load(os.path.join(save_folder, "x_coords.npy"))
    y_coords = np.load(os.path.join(save_folder, "y_coords.npy"))
    
    # 加载原始表达矩阵（如果存在）
    raw_expr_file = os.path.join(save_folder, "raw_expr_mat.npy")
    if os.path.exists(raw_expr_file):
        raw_expr_mat = np.load(raw_expr_file)
    else:
        raw_expr_mat = None
    
    # 读取MSE值
    mse = None
    info_file = os.path.join(save_folder, "training_info.txt")
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'Best MSE:' in line:
                    mse = float(line.split(':')[1].strip())
                    break
    
    if mse is None:
        mse_file = os.path.join(save_folder, "best_mse.npy")
        if os.path.exists(mse_file):
            mse = np.load(mse_file)
    
    if mse is None:
        json_file = os.path.join(save_folder, "results.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                mse = data.get('best_mse', None)
    
    if mse is None:
        print(f"  Warning: No MSE found for lambda {l}")
        mse = np.nan
    
    mse_dict[l] = mse
    best_mse_list.append(mse)
    
    # 清理数据
    expr_mat = np.nan_to_num(expr_mat, nan=0.0, posinf=0.0, neginf=0.0)
    expr_mat = np.clip(expr_mat, -1e10, 1e10)
    
    # PCA
    n_components = min(10, expr_mat.shape[1])
    expr_mat_hat = PCA(n_components=n_components).fit_transform(expr_mat)
    expr_mat_hat = np.nan_to_num(expr_mat_hat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Mclust聚类
    try:
        mclust = Mclust(rpy2.robjects.numpy2ri.numpy2rpy(expr_mat_hat), 25, "EEE")
        clustering_labels = np.array(mclust[-2])
        n_clusters = len(np.unique(clustering_labels))
        
        # 计算内部聚类指标（不需要真实标签）
        if n_clusters >= 2:
            sil_score = silhouette_score(expr_mat_hat, clustering_labels)
            db_score = davies_bouldin_score(expr_mat_hat, clustering_labels)
            ch_score = calinski_harabasz_score(expr_mat_hat, clustering_labels)
        else:
            sil_score = np.nan
            db_score = np.nan
            ch_score = np.nan
            
    except Exception as e:
        print(f"  Lambda {l} clustering failed: {e}")
        clustering_labels = None
        n_clusters = 0
        sil_score = np.nan
        db_score = np.nan
        ch_score = np.nan
    
    results[l] = {
        'expr_mat': expr_mat,
        'raw_expr_mat': raw_expr_mat,
        'gene_names': gene_names,
        'clustering_labels': clustering_labels,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'mse': mse,
        'n_clusters': n_clusters,
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'calinski_harabasz': ch_score
    }
    
    print(f"  Lambda {l}: MSE={mse:.6f}, Clusters={n_clusters}, Silhouette={sil_score:.4f}" if not np.isnan(sil_score) else f"  Lambda {l}: MSE={mse:.6f}, Clusters={n_clusters}")

# 找出最佳lambda（按MSE）
valid_mse_indices = [i for i, mse in enumerate(best_mse_list) if not np.isnan(mse)]
if valid_mse_indices:
    best_idx = valid_mse_indices[np.argmin([best_mse_list[i] for i in valid_mse_indices])]
    best_lambda_mse = l_num[best_idx]
    print(f"\n最佳lambda (按MSE): {best_lambda_mse} (MSE={best_mse_list[best_idx]:.6f})")
else:
    best_lambda_mse = l_num[0]

# 找出最佳lambda（按Silhouette）
valid_sil = [(l, results[l]['silhouette']) for l in l_num if not np.isnan(results[l]['silhouette'])]
if valid_sil:
    best_lambda_sil = max(valid_sil, key=lambda x: x[1])[0]
    print(f"最佳lambda (按Silhouette): {best_lambda_sil} (Silhouette={results[best_lambda_sil]['silhouette']:.4f})")
else:
    best_lambda_sil = best_lambda_mse

# ========== 新增模块：特定基因原始与插补后对比图 ==========
print("\n" + "="*60)
print("生成特定基因表达对比图...")
print("="*60)

# 使用最佳lambda的结果进行基因可视化
best_result = results[best_lambda_mse]
gene_names_arr = best_result['gene_names']
gene_names_lower = np.char.lower(gene_names_arr)

for gene_name in genes_to_visualize:
    # 查找基因索引（忽略大小写）
    gene_idx = np.where(gene_names_lower == gene_name.lower())[0]
    
    if len(gene_idx) == 0:
        print(f"  警告: 基因 '{gene_name}' 未找到，跳过...")
        continue
    
    gene_idx = gene_idx[0]
    actual_gene_name = gene_names_arr[gene_idx]
    
    # 获取插补后的表达值
    imputed_expr = best_result['expr_mat'][:, gene_idx]
    
    # 获取原始表达值（如果存在）
    raw_expr = None
    if best_result['raw_expr_mat'] is not None:
        raw_expr = best_result['raw_expr_mat'][:, gene_idx]
    
    # 创建对比图
    if raw_expr is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图A: 原始表达
    if raw_expr is not None:
        ax1 = axes[0]
        # 处理可能的NaN值
        raw_expr_clean = np.nan_to_num(raw_expr, nan=0.0)
        vmin_raw = np.percentile(raw_expr_clean, 5)
        vmax_raw = np.percentile(raw_expr_clean, 95)
        scatter1 = ax1.scatter(best_result['x_coords'], best_result['y_coords'], 
                               c=raw_expr_clean, cmap='RdYlBu_r', s=20, alpha=0.7,
                               vmin=vmin_raw, vmax=vmax_raw)
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_title(f'Raw Expression: {actual_gene_name}', fontsize=12)
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        plt.colorbar(scatter1, ax=ax1, label='Expression level')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # 图B: 插补后表达
        ax2 = axes[1]
    else:
        ax2 = axes[0]
    
    imputed_expr_clean = np.nan_to_num(imputed_expr, nan=0.0)
    vmin_imp = np.percentile(imputed_expr_clean, 5)
    vmax_imp = np.percentile(imputed_expr_clean, 95)
    scatter2 = ax2.scatter(best_result['x_coords'], best_result['y_coords'], 
                           c=imputed_expr_clean, cmap='RdYlBu_r', s=20, alpha=0.7,
                           vmin=vmin_imp, vmax=vmax_imp)
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_title(f'Imputed Expression: {actual_gene_name}\n(Lambda={best_lambda_mse})', fontsize=12)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    plt.colorbar(scatter2, ax=ax2, label='Expression level')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # 图C: 差值图（插补后 - 原始）
    if raw_expr is not None:
        ax3 = axes[2]
        diff_expr = imputed_expr_clean - np.nan_to_num(raw_expr, nan=0.0)
        vmin_diff = np.percentile(diff_expr, 5)
        vmax_diff = np.percentile(diff_expr, 95)
        scatter3 = ax3.scatter(best_result['x_coords'], best_result['y_coords'], 
                               c=diff_expr, cmap='RdBu_r', s=20, alpha=0.7,
                               vmin=vmin_diff, vmax=vmax_diff)
        ax3.set_ylim(ax3.get_ylim()[::-1])
        ax3.set_title(f'Difference (Imputed - Raw): {actual_gene_name}', fontsize=12)
        ax3.set_xlabel('X coordinate')
        ax3.set_ylabel('Y coordinate')
        plt.colorbar(scatter3, ax=ax3, label='Difference')
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    plt.suptitle(f'Gene {actual_gene_name} Expression: Raw vs Imputed', fontsize=14)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(gene_compare_folder, f'{actual_gene_name}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  已保存: {save_path}")
    plt.close()

# 为所有lambda生成特定基因的插补效果对比（多lambda对比）
print("\n生成多lambda插补效果对比图...")
for gene_name in genes_to_visualize:
    gene_idx = np.where(np.char.lower(results[l_num[0]]['gene_names']) == gene_name.lower())[0]
    if len(gene_idx) == 0:
        continue
    
    actual_gene_name = results[l_num[0]]['gene_names'][gene_idx[0]]
    
    # 创建2x4的子图（8个lambda）
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, l in enumerate(l_num):
        ax = axes[idx]
        res = results[l]
        
        expr = res['expr_mat'][:, gene_idx[0]]
        expr_clean = np.nan_to_num(expr, nan=0.0)
        vmin = np.percentile(expr_clean, 5)
        vmax = np.percentile(expr_clean, 95)
        
        scatter = ax.scatter(res['x_coords'], res['y_coords'], 
                            c=expr_clean, cmap='RdYlBu_r', s=10, alpha=0.7,
                            vmin=vmin, vmax=vmax)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_title(f'Lambda={l}\nMSE={res["mse"]:.5f}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Gene {actual_gene_name} Imputation Results for Different Lambda', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(gene_compare_folder, f'{actual_gene_name}_all_lambdas.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  已保存: {save_path}")
    plt.close()

# ========== 保存分析结果到文本文件 ==========
with open(os.path.join(figures_folder, "analysis_summary.txt"), 'w') as f:
    f.write("="*60 + "\n")
    f.write("GNTD Spatial Transcriptomics Imputation Results\n")
    f.write("="*60 + "\n\n")
    f.write("Evaluation Metrics (without ground truth labels):\n")
    f.write("- MSE: Lower is better (imputation accuracy)\n")
    f.write("- Silhouette Score: Higher is better (cluster cohesion/separation)\n")
    f.write("- Davies-Bouldin Score: Lower is better (cluster similarity)\n")
    f.write("- Calinski-Harabasz Score: Higher is better (cluster variance ratio)\n\n")
    
    for l in l_num:
        f.write(f"Lambda = {l}:\n")
        f.write(f"  MSE: {results[l]['mse']:.6f}\n" if not np.isnan(results[l]['mse']) else f"  MSE: N/A\n")
        f.write(f"  Number of Clusters: {results[l]['n_clusters']}\n")
        if not np.isnan(results[l]['silhouette']):
            f.write(f"  Silhouette Score: {results[l]['silhouette']:.4f}\n")
            f.write(f"  Davies-Bouldin Score: {results[l]['davies_bouldin']:.4f}\n")
            f.write(f"  Calinski-Harabasz Score: {results[l]['calinski_harabasz']:.2f}\n")
        f.write("\n")
    
    f.write(f"\n最佳lambda (按MSE): {best_lambda_mse}\n")
    f.write(f"最佳MSE: {results[best_lambda_mse]['mse']:.6f}\n")
    if valid_sil:
        f.write(f"\n最佳lambda (按Silhouette): {best_lambda_sil}\n")
        f.write(f"最佳Silhouette: {results[best_lambda_sil]['silhouette']:.4f}\n")
    
    f.write(f"\n\n可视化基因列表: {genes_to_visualize}\n")
    f.write(f"基因对比图保存位置: {gene_compare_folder}\n")

print(f"\n分析结果已保存到: {figures_folder}/analysis_summary.txt")

# ========== 原有绘图代码保持不变 ==========
# 图1: MSE vs Lambda
fig, ax = plt.subplots(figsize=(10, 6))
valid_l = [l for l, mse in mse_dict.items() if not np.isnan(mse)]
valid_mse = [mse_dict[l] for l in valid_l]
ax.plot(valid_l, valid_mse, 'o-', linewidth=2, markersize=8, color='blue')
ax.set_xscale('log')
ax.set_xlabel('Lambda (log scale)', fontsize=14)
ax.set_ylabel('Best MSE (lower is better)', fontsize=14)
ax.set_title('MSE vs Regularization Weight', fontsize=16)
ax.grid(True, alpha=0.3)
for l, mse in zip(valid_l, valid_mse):
    ax.annotate(f'{mse:.5f}', (l, mse), xytext=(5, 5), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'mse_vs_lambda.png'), dpi=150)
print(f"图1已保存: {figures_folder}/mse_vs_lambda.png")
plt.close()

# 图2: Silhouette Score vs Lambda
fig, ax = plt.subplots(figsize=(10, 6))
sil_scores = [results[l]['silhouette'] for l in l_num]
ax.plot(l_num, sil_scores, 'o-', linewidth=2, markersize=8, color='green')
ax.set_xscale('log')
ax.set_xlabel('Lambda (log scale)', fontsize=14)
ax.set_ylabel('Silhouette Score (higher is better)', fontsize=14)
ax.set_title('Cluster Quality vs Regularization Weight', fontsize=16)
ax.grid(True, alpha=0.3)
for l, sil in zip(l_num, sil_scores):
    if not np.isnan(sil):
        ax.annotate(f'{sil:.4f}', (l, sil), xytext=(5, 5), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'silhouette_vs_lambda.png'), dpi=150)
print(f"图2已保存: {figures_folder}/silhouette_vs_lambda.png")
plt.close()

# 图3: 聚类数量 vs Lambda
fig, ax = plt.subplots(figsize=(10, 6))
n_clusters_list = [results[l]['n_clusters'] for l in l_num]
ax.plot(l_num, n_clusters_list, 'o-', linewidth=2, markersize=8, color='red')
ax.set_xscale('log')
ax.set_xlabel('Lambda (log scale)', fontsize=14)
ax.set_ylabel('Number of Clusters', fontsize=14)
ax.set_title('Number of Detected Clusters vs Regularization Weight', fontsize=16)
ax.grid(True, alpha=0.3)
for l, n in zip(l_num, n_clusters_list):
    ax.annotate(f'{n}', (l, n), xytext=(5, 5), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'n_clusters_vs_lambda.png'), dpi=150)
print(f"图3已保存: {figures_folder}/n_clusters_vs_lambda.png")
plt.close()

# 图4: 最佳lambda的聚类可视化（按MSE）
best_result = results[best_lambda_mse]
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(best_result['x_coords'], best_result['y_coords'], 
                     c=best_result['clustering_labels'], cmap='tab20', s=20, alpha=0.7)
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_title(f'Clustering Results (Best by MSE: Lambda={best_lambda_mse})\nMSE={best_result["mse"]:.6f}, {best_result["n_clusters"]} clusters', fontsize=12)
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, f'clustering_best_mse.png'), dpi=150)
print(f"图4已保存: {figures_folder}/clustering_best_mse.png")
plt.close()

# 图5: 最佳Silhouette的聚类可视化
if valid_sil:
    best_sil_result = results[best_lambda_sil]
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(best_sil_result['x_coords'], best_sil_result['y_coords'], 
                         c=best_sil_result['clustering_labels'], cmap='tab20', s=20, alpha=0.7)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title(f'Clustering Results (Best by Silhouette: Lambda={best_lambda_sil})\nSilhouette={best_sil_result["silhouette"]:.4f}, {best_sil_result["n_clusters"]} clusters', fontsize=12)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, f'clustering_best_silhouette.png'), dpi=150)
    print(f"图5已保存: {figures_folder}/clustering_best_silhouette.png")
    plt.close()

# 图6: 不同lambda的聚类结果对比（所有lambda）
fig, axes = plt.subplots(2, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, l in enumerate(l_num):
    if results[l]['clustering_labels'] is not None:
        ax = axes[idx]
        res = results[l]
        scatter = ax.scatter(res['x_coords'], res['y_coords'], 
                            c=res['clustering_labels'], cmap='tab20', s=10, alpha=0.7)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_title(f'Lambda={l}\nMSE={res["mse"]:.5f}\n{res["n_clusters"]} clusters', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Clustering Results for Different Lambda Values', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'clustering_comparison_all.png'), dpi=150)
print(f"图6已保存: {figures_folder}/clustering_comparison_all.png")
plt.close()

# 图7: 选择4个代表性lambda的详细对比
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
lambdas_to_show = [0, 0.01, 0.1, 1]
axes = axes.flatten()

for idx, l in enumerate(lambdas_to_show):
    if l in results and results[l]['clustering_labels'] is not None:
        ax = axes[idx]
        res = results[l]
        scatter = ax.scatter(res['x_coords'], res['y_coords'], 
                            c=res['clustering_labels'], cmap='tab20', s=15, alpha=0.7)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_title(f'Lambda={l}\nMSE={res["mse"]:.5f}, {res["n_clusters"]} clusters', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Clustering Results for Selected Lambda Values', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'clustering_comparison_selected.png'), dpi=150)
print(f"图7已保存: {figures_folder}/clustering_comparison_selected.png")
plt.close()

# 图8: MSE柱状图
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([str(l) for l in valid_l], valid_mse, color='steelblue', alpha=0.7)
ax.set_xlabel('Lambda', fontsize=14)
ax.set_ylabel('Best MSE (lower is better)', fontsize=14)
ax.set_title('MSE for Different Lambda Values', fontsize=16)
ax.grid(True, alpha=0.3, axis='y')
for i, (l, mse) in enumerate(zip(valid_l, valid_mse)):
    ax.text(i, mse, f'{mse:.5f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'mse_bar_chart.png'), dpi=150)
print(f"图8已保存: {figures_folder}/mse_bar_chart.png")
plt.close()

# 打印总结
print("\n" + "="*60)
print("分析完成！")
print("="*60)
print(f"最佳lambda (按MSE): {best_lambda_mse}")
print(f"  - MSE: {results[best_lambda_mse]['mse']:.6f}")
print(f"  - 聚类数量: {results[best_lambda_mse]['n_clusters']}")
if valid_sil:
    print(f"\n最佳lambda (按Silhouette): {best_lambda_sil}")
    print(f"  - Silhouette: {results[best_lambda_sil]['silhouette']:.4f}")
    print(f"  - 聚类数量: {results[best_lambda_sil]['n_clusters']}")
print(f"\n所有结果已保存到 '{figures_folder}' 文件夹:")
print(f"  - analysis_summary.txt (分析结果汇总)")
print(f"  - mse_vs_lambda.png (MSE随lambda变化图)")
print(f"  - silhouette_vs_lambda.png (Silhouette随lambda变化图)")
print(f"  - n_clusters_vs_lambda.png (聚类数量随lambda变化图)")
print(f"  - mse_bar_chart.png (MSE柱状图)")
print(f"  - clustering_best_mse.png (最佳MSE lambda聚类图)")
print(f"  - clustering_best_silhouette.png (最佳Silhouette lambda聚类图)")
print(f"  - clustering_comparison_all.png (所有lambda聚类对比)")
print(f"  - clustering_comparison_selected.png (代表性lambda聚类对比)")
print(f"\n基因表达对比图已保存到 '{gene_compare_folder}' 文件夹:")
print(f"  - [基因名]_comparison.png (原始 vs 插补 vs 差值)")
print(f"  - [基因名]_all_lambdas.png (不同lambda的插补效果对比)")
print("="*60)