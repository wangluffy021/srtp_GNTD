import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import warnings

warnings.filterwarnings('ignore')

# ========================== 配置 ==========================
results_dir = "results_GNTD_mouse_ARI"
figures_dir = "figures_ARI"
os.makedirs(figures_dir, exist_ok=True)

# 需要可视化的基因（可自行增删）
genes_to_visualize = ['gfap', 'syn1', 'mbp', 'nefh', 'lamp2']
target_tucker_rank = (48, 43, 43)

print("正在加载 Tucker 分解实验结果...\n")


def normalize_tucker_rank(rank):
    if isinstance(rank, int):
        return (rank, rank, rank)

    rank = tuple(int(v) for v in rank)
    if len(rank) != 3:
        raise ValueError("Tucker rank must contain exactly three integers")

    return rank


def format_tucker_rank(rank):
    rank_g, rank_x, rank_y = normalize_tucker_rank(rank)
    return f"L={rank_g}, M={rank_x}, N={rank_y}"


def extract_tucker_rank(data):
    rank_data = data.get('tucker_rank', data.get('rank'))
    if rank_data is None:
        raise KeyError("Result file does not contain rank or tucker_rank metadata")

    rank_array = np.asarray(rank_data).astype(int).flatten()
    if rank_array.size == 1:
        rank = int(rank_array.item())
        return (rank, rank, rank)
    if rank_array.size >= 3:
        return tuple(int(v) for v in rank_array[:3])

    raise ValueError("Unable to parse Tucker rank from result file")


def matlab_to_str(value):
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"}:
        return "".join(array.flatten().tolist()).strip()
    if array.size == 1:
        return str(array.item()).strip()
    return str(value).strip()


# ====================== 1. 收集所有实验结果 ======================
mat_files = [f for f in os.listdir(results_dir) if f.endswith('.mat')]

if not mat_files:
    print("错误：results_GNTD_mouse_ARI 文件夹中没有 .mat 文件")
    exit()

results = []
for f in mat_files:
    data = loadmat(os.path.join(results_dir, f))
    rank = extract_tucker_rank(data)
    if target_tucker_rank is not None and rank != normalize_tucker_rank(target_tucker_rank):
        continue

    lam = float(data['lambda'].item())
    ari = float(data.get('ARI', np.nan).item())
    mse = float(data['best_mse'].item())
    results.append({
        'lambda': lam,
        'ARI': ari,
        'MSE': mse,
        'rank': rank,
        'file': f,
        'data': data,
    })

if not results:
    if target_tucker_rank is None:
        print("错误：未找到可用的 Tucker 实验结果")
    else:
        print(f"错误：未找到 Tucker rank = {normalize_tucker_rank(target_tucker_rank)} 的结果文件")
    exit()

rank_groups = {}
for result in results:
    rank_groups.setdefault(result['rank'], []).append(result)

for entries in rank_groups.values():
    entries.sort(key=lambda x: x['lambda'])

valid_ari_results = [r for r in results if not np.isnan(r['ARI'])]
if valid_ari_results:
    best_result = max(valid_ari_results, key=lambda x: (x['ARI'], -x['MSE']))
else:
    best_result = min(results, key=lambda x: x['MSE'])

best_lam = best_result['lambda']
best_ari = best_result['ARI']
best_mse = best_result['MSE']
best_rank = best_result['rank']
best_rank_label = format_tucker_rank(best_rank)
best_data = best_result['data']

print(f"最佳模型 → λ={best_lam}, Tucker rank=({best_rank_label}), ARI={best_ari:.4f}, MSE={best_mse:.5f}\n")

# ====================== 2. 绘制 ARI 和 MSE 折线图 ======================
print("正在生成 ARI 和 MSE vs Lambda 折线图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ARI / MSE 图
for rank in sorted(rank_groups.keys()):
    group = rank_groups[rank]
    lambdas = [r['lambda'] for r in group]
    aris = [r['ARI'] for r in group]
    mses = [r['MSE'] for r in group]
    rank_label = format_tucker_rank(rank)

    axes[0].plot(lambdas, aris, 'o-', linewidth=2, markersize=6, label=rank_label)
    axes[1].plot(lambdas, mses, 'o-', linewidth=2, markersize=6, label=rank_label)

axes[0].set_xlabel('Lambda (λ)', fontsize=12)
axes[0].set_ylabel('Adjusted Rand Index (ARI)', fontsize=12)
axes[0].set_title('ARI vs Lambda', fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()

axes[1].set_xlabel('Lambda (λ)', fontsize=12)
axes[1].set_ylabel('Validation MSE', fontsize=12)
axes[1].set_title('MSE vs Lambda', fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend()

plt.suptitle(
    f'GNTD Mouse Tissue | Best ARI = {best_ari:.4f} at λ={best_lam}, ({best_rank_label})',
    fontsize=15,
)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'ARI_MSE_vs_Lambda_Tucker.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"已保存: ARI_MSE_vs_Lambda_Tucker.png\n")

# ====================== 3. 使用最佳 lambda 生成 Raw vs Imputed 对比图（仅两图） ======================
print(f"使用最佳 λ={best_lam} 和 Tucker rank=({best_rank_label}) 生成基因表达对比图（Raw vs Imputed）...\n")

raw_genes = best_data['gene_names']
gene_names_original = [matlab_to_str(g) for g in raw_genes.flatten()]
gene_names_lower = [g.lower() for g in gene_names_original]

expr_imputed = best_data['expr_mat']
expr_raw = best_data.get('expr_raw_mat')
x_coords = best_data['x_coords'].flatten()
y_coords = best_data['y_coords'].flatten()
clustering_labels = np.asarray(best_data.get('clustering_labels', np.array([]))).flatten()
ground_truth = np.asarray(best_data.get('ground_truth', np.array([]))).flatten()

# ====================== 3. 生成最佳模型聚类图 ======================
if clustering_labels.size == x_coords.size:
    print("正在生成最佳模型聚类图...\n")

    if ground_truth.size == x_coords.size:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        sc = ax.scatter(x_coords, y_coords, c=ground_truth, cmap='tab20', s=28, alpha=0.9)
        ax.set_title('Ground Truth Clusters')
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Cluster')

        ax = axes[1]
        sc = ax.scatter(x_coords, y_coords, c=clustering_labels, cmap='tab20', s=28, alpha=0.9)
        ax.set_title(f'Predicted Clusters ({best_rank_label}, λ={best_lam})')
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Cluster')

        plt.suptitle(f'Best Clustering Result | ARI={best_ari:.4f}', fontsize=14)
        plt.tight_layout()
        cluster_fig_name = 'Best_Clustering_vs_GroundTruth.png'
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(x_coords, y_coords, c=clustering_labels, cmap='tab20', s=28, alpha=0.9)
        ax.set_title(f'Predicted Clusters ({best_rank_label}, λ={best_lam})')
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Cluster')
        plt.tight_layout()
        cluster_fig_name = 'Best_Clustering_Result.png'

    plt.savefig(os.path.join(figures_dir, cluster_fig_name), dpi=220, bbox_inches='tight')
    plt.close()
    print(f" 已保存: {cluster_fig_name}\n")
else:
    print("警告：结果文件中未找到可用的 clustering_labels，跳过聚类图生成。\n")

for gene_lower in genes_to_visualize:
    if gene_lower not in gene_names_lower:
        print(f"  跳过：基因 {gene_lower.upper()} 未找到")
        continue
    
    idx = gene_names_lower.index(gene_lower)
    actual_name = gene_names_original[idx]
    
    imp_clean = np.nan_to_num(expr_imputed[:, idx], nan=0.0)
    
    # 只绘制 Raw 和 Imputed 两张图
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Raw 图
    if expr_raw is not None and expr_raw.shape == expr_imputed.shape:
        raw_clean = np.nan_to_num(expr_raw[:, idx], nan=0.0)
        ax = axes[0]
        vmin, vmax = np.percentile(raw_clean[raw_clean > 0], [5, 95]) if np.any(raw_clean > 0) else (0, raw_clean.max())
        sc = ax.scatter(x_coords, y_coords, c=raw_clean, cmap='RdYlBu_r', s=28, alpha=0.85, vmin=vmin, vmax=vmax)
        ax.set_title(f'Raw: {actual_name.upper()}')
        plt.colorbar(sc, ax=ax, label='Expression')
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
    else:
        print(f"  警告：Raw 数据不可用，仅显示 Imputed 图")
        ax = axes[0]
        ax.set_title(f'Raw: {actual_name.upper()} (Not Available)')
        ax.text(0.5, 0.5, 'Raw Data\nNot Available', ha='center', va='center', transform=ax.transAxes)
    
    # Imputed 图
    ax = axes[1]
    vmin, vmax = np.percentile(imp_clean[imp_clean > 0], [5, 95]) if np.any(imp_clean > 0) else (0, imp_clean.max())
    sc = ax.scatter(x_coords, y_coords, c=imp_clean, cmap='RdYlBu_r', s=28, alpha=0.85, vmin=vmin, vmax=vmax)
    ax.set_title(f'Imputed (λ={best_lam}, {best_rank_label})')
    plt.colorbar(sc, ax=ax, label='Expression')
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        f'{actual_name.upper()} | Best λ={best_lam} | {best_rank_label} | ARI={best_ari:.4f} | MSE={best_mse:.5f}',
        fontsize=14,
    )
    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, f'{actual_name.upper()}_BestLambda_Raw_vs_Imputed.png')
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f" 已保存: {actual_name.upper()}_BestLambda_Raw_vs_Imputed.png")

# ====================== 总结 ======================
print(f"\n🎉 所有可视化完成！")
print(f"最佳 ARI = {best_ari:.4f} （λ={best_lam}, {best_rank_label}）")
print(f"折线图保存在：{figures_dir}/ARI_MSE_vs_Lambda_Tucker.png")
print(f"聚类图保存在：{figures_dir}/ 文件夹")
print(f"基因对比图保存在：{figures_dir}/ 文件夹（仅 Raw + Imputed）")
print(f"推荐重点查看：GFAP、SYN1、MBP 的对比图")
