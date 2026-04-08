# running.py - 多参数循环最终版（推荐使用）

from GNTD import GNTD
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import numpy as np
import warnings
from scipy.io import savemat

warnings.filterwarnings('ignore')

# ========================== 配置 ==========================
raw_data_path = "/home/wangluffy/projects/GNTD/data/tissue"
PPI_data_path = "/home/wangluffy/projects/GNTD/data/BIOGRID-ORGANISM-Mus_musculus-4.4.209.tab3.txt"

rank_list = [128]
l_list   = [0.08,0.09,0.1,0.11,0.12,0.13]

output_dir = "results_GNTD_mouse_ARI"
os.makedirs(output_dir, exist_ok=True)

print(f"即将运行 {len(l_list)} × {len(rank_list)} = {len(l_list)*len(rank_list)} 个实验\n")

# =========================================================

for l in l_list:
    for rank in rank_list:
        print(f"\n{'='*85}")
        print(f"正在运行： lambda = {l} ,  rank = {rank}")
        print(f"{'='*85}")

        model = GNTD(raw_data_path, PPI_data_path)

        print("开始预处理...")
        model.preprocess(use_coexpression=False, n_top_genes=3000, load_labels=True)
        print("预处理完成。\n")

        print("开始插补训练...")
        best_mse = model.impute(rank=rank, l=l, lr=0.003, max_epoch=3000, verbose=True)

        expr_mat, gene_names = model.get_imputed_expr_mat()
        expr_raw_mat, _ = model.get_raw_expr_mat()
        x_coords, y_coords = model.get_sp_coords()

        # ====================== ARI 计算 ======================
        spot_idx = np.where(model.mapping[:, -1] != -2)[0]
        ground_truth = model.mapping[spot_idx, -1].astype(int)

        n_clusters = len(np.unique(ground_truth[ground_truth >= 0]))
        pca = PCA(n_components=min(20, expr_mat.shape[1]), random_state=42)
        expr_pca = pca.fit_transform(expr_mat)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clustering_labels = kmeans.fit_predict(expr_pca)

        ari_value = adjusted_rand_score(ground_truth, clustering_labels)
        print(f"ARI: {ari_value:.5f}   |   MSE: {best_mse:.5f}")

        # ====================== 保存结果 ======================
        savefile_name = os.path.join(
            output_dir,
            f"GNTD_l{l}_r{rank}_MSE{best_mse:.5f}_ARI{ari_value:.5f}.mat"
        )

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
            "rank": rank
        })

        print(f"✅ 保存完成：{savefile_name}\n")

print("\n🎉 所有实验运行完毕！")
print(f"共生成 {len(l_list)*len(rank_list)} 个结果文件，保存在 {output_dir}/ 文件夹")