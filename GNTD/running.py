# GNTD_NEED_DATA_simple.py (基于你的原文件最小修改)
from GNTD import GNTD
from sklearn.metrics import adjusted_rand_score  # 导入 ARI 计算函数
import os
import numpy as np
from scipy.io import loadmat, savemat
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

raw_data_path = "/home/wangluffy/projects/GNTD/data/tissue"  # Path to spatial expression data
PPI_data_path = "/home/wangluffy/projects/GNTD/data/BIOGRID-ORGANISM-Mus_musculus-4.4.209.tab3.txt"  # PPI data

rank = 128 # tensor rank
l_num = [0, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]

# ========== 修改点1: 将循环移到正确位置 ==========
for l in l_num:
    print(f"\n{'='*50}")
    print(f"Processing lambda = {l}")
    print(f"{'='*50}")
    
    model = GNTD(raw_data_path, PPI_data_path) # GNTD class initialization
    print("Model initialized.")
    
    print("Starting preprocessing...")
    model.preprocess(use_coexpression=False, n_top_genes=3000)
    print("Preprocessing completed.")
    
    print("Starting imputation...")
    best_mse = model.impute(rank, l, lr=0.005, max_epoch=3000)
    print(f"Imputation completed. Best MSE: {best_mse}")
    
    expr_mat, gene_names = model.get_imputed_expr_mat() # Get imputed expression matrix
    print("Imputation results retrieved.")
    expr_raw_mat = model.get_raw_expr_mat()
    
    x_coords, y_coords = model.get_sp_coords() # Get spatial coordinates
    
    # ========== 修改点2: 为每个lambda创建独立文件夹保存结果 ==========
    save_folder = f"GNTD_results_lambda_{l}"
    os.makedirs(save_folder, exist_ok=True)
    
    # Save the imputed expression matrix, gene names, and spatial coordinates
    np.save(os.path.join(save_folder, "imputed_expr_mat.npy"), expr_mat)
    np.save(os.path.join(save_folder, "gene_names.npy"), gene_names)
    np.save(os.path.join(save_folder, "x_coords.npy"), x_coords)
    np.save(os.path.join(save_folder, "y_coords.npy"), y_coords)
    np.save(os.path.join(save_folder, "raw_expr_mat.npy"), expr_raw_mat[0])  # 保存原始表达矩阵
    np.save(os.path.join(save_folder, "mapping.npy"), model.mapping)  # 保存映射信息
    
    # 保存训练信息
    with open(os.path.join(save_folder, "training_info.txt"), 'w') as f:
        f.write(f"Lambda: {l}\n")
        f.write(f"Rank: {rank}\n")
        f.write(f"Best MSE: {best_mse}\n")
        f.write(f"Imputed matrix shape: {expr_mat.shape}\n")
        f.write(f"Number of genes: {len(gene_names)}\n")
        f.write(f"Number of spots: {len(x_coords)}\n")
    
    print(f"Results saved to {save_folder}/")
    print(f"  - imputed_expr_mat.npy: {expr_mat.shape}")
    print(f"  - gene_names.npy: {len(gene_names)} genes")
    print(f"  - x_coords.npy, y_coords.npy: {len(x_coords)} spots")

print("\n" + "="*50)
print("All training completed!")
print("="*50)