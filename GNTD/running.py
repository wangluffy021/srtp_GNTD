# running.py - 多参数循环最终版（推荐使用）

import os
import random
import hashlib
import warnings

# 需要在导入 torch 之前设置，提升 CUDA 计算可复现性
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.io import savemat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from GNTD import GNTD

warnings.filterwarnings('ignore')


def set_global_determinism(seed: int = 42):
    """固定所有常见随机源，尽量保证每次运行结果一致。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # 若遇到不支持的确定性算子会报错，便于显式发现不稳定来源
    torch.use_deterministic_algorithms(True)


def normalize_tucker_rank(rank):
    if isinstance(rank, int):
        if rank <= 0:
            raise ValueError("rank must be a positive integer")
        return (rank, rank, rank)

    if not isinstance(rank, (tuple, list)) or len(rank) != 3:
        raise ValueError("rank must be an int or a tuple/list with three Tucker ranks (L, M, N)")

    rank = tuple(int(v) for v in rank)
    if min(rank) <= 0:
        raise ValueError("all Tucker ranks must be positive integers")

    return rank


def format_tucker_rank(rank) -> str:
    rank_g, rank_x, rank_y = normalize_tucker_rank(rank)
    return f"L{rank_g}_M{rank_x}_N{rank_y}"


def seed_from_config(rank, l_value: float, base_seed: int = 42) -> int:
    """为每组参数生成稳定种子，避免实验顺序影响结果。"""
    rank = normalize_tucker_rank(rank)
    key = f"{base_seed}-{rank}-{l_value:.8f}".encode("utf-8")
    hashed = int(hashlib.md5(key).hexdigest()[:8], 16)
    return (base_seed + hashed) % (2**31 - 1)


# ========================== 配置 ==========================
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = "/home/wangluffy/projects/GNTD/data/tissue"
PPI_data_path = "/home/wangluffy/projects/GNTD/data/BIOGRID-ORGANISM-Mus_musculus-4.4.209.tab3.txt"

# Final selected Tucker parameter setting based on ARI search results.
tucker_rank_list = [
    (48, 43, 43),
]

# Final selected graph regularization weight.
l_list = [0.009]

base_seed = 42
clustering_seed = 42
output_dir = os.path.join(script_dir, "results_GNTD_mouse_ARI")
os.makedirs(output_dir, exist_ok=True)

# 先固定一次全局随机性
set_global_determinism(base_seed)

print(f"即将运行 {len(l_list)} × {len(tucker_rank_list)} = {len(l_list)*len(tucker_rank_list)} 个实验\n")

# =========================================================

model = GNTD(raw_data_path, PPI_data_path)
print("开始预处理（仅执行一次）...")
model.preprocess(use_coexpression=False, n_top_genes=3000, load_labels=True)
print("预处理完成。\n")

for l in l_list:
    for tucker_rank in tucker_rank_list:
        tucker_rank = normalize_tucker_rank(tucker_rank)
        rank_label = format_tucker_rank(tucker_rank)
        run_seed = seed_from_config(rank=tucker_rank, l_value=l, base_seed=base_seed)
        set_global_determinism(run_seed)

        print(f"\n{'='*85}")
        print(f"正在运行： lambda = {l} , Tucker rank = {tucker_rank} , seed = {run_seed}")
        print(f"{'='*85}")

        print("开始插补训练...")
        best_mse = model.impute(rank=tucker_rank, l=l, lr=0.003, max_epoch=3000, verbose=True)

        expr_mat, gene_names = model.get_imputed_expr_mat()
        expr_raw_mat, _ = model.get_raw_expr_mat()
        x_coords, y_coords = model.get_sp_coords()

        # ====================== ARI 计算 ======================
        spot_idx = np.where(model.mapping[:, -1] != -2)[0]
        ground_truth = model.mapping[spot_idx, -1].astype(int)

        n_clusters = len(np.unique(ground_truth[ground_truth >= 0]))
        pca = PCA(n_components=min(20, expr_mat.shape[1]), random_state=clustering_seed)
        expr_pca = pca.fit_transform(expr_mat)

        # Keep clustering randomness fixed across all parameter settings so
        # ARI differences reflect model changes rather than KMeans noise.
        kmeans = KMeans(n_clusters=n_clusters, random_state=clustering_seed, n_init=20)
        clustering_labels = kmeans.fit_predict(expr_pca)

        ari_value = adjusted_rand_score(ground_truth, clustering_labels)
        print(f"ARI: {ari_value:.5f}   |   MSE: {best_mse:.5f}")

        # ====================== 保存结果 ======================
        savefile_name = os.path.join(
            output_dir,
            f"GNTD_l{l}_{rank_label}_S{run_seed}_MSE{best_mse:.5f}_ARI{ari_value:.5f}.mat"
        )

        # 兜底：防止执行过程中工作目录变化导致相对路径失效
        os.makedirs(output_dir, exist_ok=True)
        savemat(savefile_name, {
            "expr_mat": expr_mat,
            "expr_raw_mat": expr_raw_mat,
            "gene_names": gene_names,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "mapping": model.mapping,
            "clustering_labels": clustering_labels,
            "ground_truth": ground_truth,
            "ARI": ari_value,
            "best_mse": best_mse,
            "lambda": l,
            "rank": np.array(tucker_rank, dtype=np.int32),
            "tucker_rank": np.array(tucker_rank, dtype=np.int32),
            "seed": run_seed,
            "clustering_seed": clustering_seed,
        })

        print(f"✅ 保存完成：{savefile_name}\n")

print("\n🎉 所有实验运行完毕！")
print(f"共生成 {len(l_list)*len(tucker_rank_list)} 个结果文件，保存在 {output_dir}/ 文件夹")
