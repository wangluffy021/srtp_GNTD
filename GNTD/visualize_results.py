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

print("正在加载所有 rank=128 的实验结果...\n")

# ====================== 1. 收集所有 rank=128 的结果 ======================
mat_files = [f for f in os.listdir(results_dir) if f.endswith('.mat') and '_r128_' in f]

if not mat_files:
    print("错误：results_GNTD_mouse_ARI 文件夹中没有 rank=128 的 .mat 文件")
    exit()

results = []
for f in mat_files:
    data = loadmat(os.path.join(results_dir, f))
    lam = float(data['lambda'].item())
    ari = float(data.get('ARI', np.nan).item())
    mse = float(data['best_mse'].item())
    results.append({'lambda': lam, 'ARI': ari, 'MSE': mse, 'file': f, 'data': data})

# 按 lambda 排序
results.sort(key=lambda x: x['lambda'])

lambdas = [r['lambda'] for r in results]
aris = [r['ARI'] for r in results]
mses = [r['MSE'] for r in results]

# 找到最佳 lambda（ARI 最高）
best_idx = np.argmax(aris)
best_lam = lambdas[best_idx]
best_ari = aris[best_idx]
best_mse = mses[best_idx]
best_data = results[best_idx]['data']

print(f"最佳模型 → λ={best_lam}, rank=128, ARI={best_ari:.4f}, MSE={best_mse:.5f}\n")

# ====================== 2. 绘制 ARI 和 MSE 折线图 ======================
print("正在生成 ARI 和 MSE vs Lambda 折线图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ARI 图
axes[0].plot(lambdas, aris, 'o-', color='tab:blue', linewidth=2, markersize=8, label='ARI')
axes[0].set_xlabel('Lambda (λ)', fontsize=12)
axes[0].set_ylabel('Adjusted Rand Index (ARI)', fontsize=12)
axes[0].set_title('ARI vs Lambda (rank=128)', fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()

# MSE 图
axes[1].plot(lambdas, mses, 'o-', color='tab:red', linewidth=2, markersize=8, label='MSE')
axes[1].set_xlabel('Lambda (λ)', fontsize=12)
axes[1].set_ylabel('Validation MSE', fontsize=12)
axes[1].set_title('MSE vs Lambda (rank=128)', fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend()

plt.suptitle(f'GNTD Mouse Tissue | rank=128 | Best ARI = {best_ari:.4f} at λ={best_lam}', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'ARI_MSE_vs_Lambda_rank128.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"已保存: ARI_MSE_vs_Lambda_rank128.png\n")

# ====================== 3. 使用最佳 lambda 生成 Raw vs Imputed 对比图（仅两图） ======================
print(f"使用最佳 λ={best_lam} 的结果生成基因表达对比图（Raw vs Imputed）...\n")

raw_genes = best_data['gene_names']
gene_names_lower = [str(g).strip().lower() for g in raw_genes.flatten()]
gene_names_original = [str(g).strip() for g in raw_genes.flatten()]

expr_imputed = best_data['expr_mat']
expr_raw = best_data.get('expr_raw_mat')
x_coords = best_data['x_coords'].flatten()
y_coords = best_data['y_coords'].flatten()

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
    ax.set_title(f'Imputed (λ={best_lam}, rank=128)')
    plt.colorbar(sc, ax=ax, label='Expression')
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f'{actual_name.upper()} | Best λ={best_lam} | ARI={best_ari:.4f} | MSE={best_mse:.5f}', 
                 fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, f'{actual_name.upper()}_BestLambda_Raw_vs_Imputed.png')
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f" 已保存: {actual_name.upper()}_BestLambda_Raw_vs_Imputed.png")

# ====================== 总结 ======================
print(f"\n🎉 所有可视化完成！")
print(f"最佳 ARI = {best_ari:.4f} （λ={best_lam}, rank=128）")
print(f"折线图保存在：{figures_dir}/ARI_MSE_vs_Lambda_rank128.png")
print(f"基因对比图保存在：{figures_dir}/ 文件夹（仅 Raw + Imputed）")
print(f"推荐重点查看：GFAP、SYN1、MBP 的对比图")