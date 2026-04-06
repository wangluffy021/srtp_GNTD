# check_environment.py
import sys
import subprocess
import importlib
import numpy as np
import pandas as pd

def check_python_packages():
    """检查Python包"""
    print("="*50)
    print("检查Python包...")
    print("="*50)
    
    required_packages = {
        'scanpy': '1.9.0',
        'anndata': '0.8.0',
        'numpy': '1.21.0',
        'scipy': '1.7.0',
        'pandas': '1.3.0',
        'scikit-learn': '0.24.0',
        'matplotlib': '3.4.0',
        'seaborn': '0.11.0',
        'torch': '1.9.0',
        'rpy2': '3.4.0'  # 用于R交互
    }
    
    missing_packages = []
    version_issues = []
    
    for package, min_version in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
            
            # 检查版本（简化版）
            if version != 'unknown' and version < min_version:
                version_issues.append(f"{package} {version} < {min_version}")
        except ImportError:
            print(f"✗ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺失的包: {missing_packages}")
        print("安装命令: pip install " + " ".join(missing_packages))
    
    if version_issues:
        print(f"\n版本问题: {version_issues}")
    
    return len(missing_packages) == 0

def check_r_and_mclust():
    """检查R和mclust"""
    print("\n" + "="*50)
    print("检查R和mclust...")
    print("="*50)
    
    # 检查R是否安装
    try:
        result = subprocess.run(['R', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ R已安装: {version_line}")
        else:
            print("✗ R未找到")
            return False
    except FileNotFoundError:
        print("✗ R未安装或不在PATH中")
        return False
    
    # 检查mclust包
    try:
        # 创建R脚本检查mclust
        r_script = """
        if (!require(mclust, quietly = TRUE)) {
            cat("FALSE")
        } else {
            cat(paste("TRUE", packageVersion("mclust")))
        }
        """
        
        result = subprocess.run(['R', '--slave', '-e', r_script], 
                               capture_output=True, text=True)
        
        if result.stdout.strip().startswith("TRUE"):
            version = result.stdout.strip().split()[1]
            print(f"✓ mclust已安装: 版本 {version}")
            return True
        else:
            print("✗ mclust未安装")
            print("\n安装mclust的方法:")
            print("在R中运行:")
            print('  install.packages("mclust")')
            return False
    except Exception as e:
        print(f"✗ 检查mclust时出错: {e}")
        return False

def check_rpy2():
    """检查rpy2连接"""
    print("\n" + "="*50)
    print("检查rpy2连接...")
    print("="*50)
    
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        
        # 测试基本R功能
        version = ro.r('R.version.string')[0]
        print(f"✓ rpy2可以连接R: {version}")
        
        # 测试导入mclust
        try:
            mclust = importr('mclust')
            print("✓ rpy2可以导入mclust")
            return True
        except Exception as e:
            print(f"✗ rpy2无法导入mclust: {e}")
            return False
    except ImportError:
        print("✗ rpy2未安装")
        print("安装命令: pip install rpy2")
        return False
    except Exception as e:
        print(f"✗ rpy2连接失败: {e}")
        return False

def check_data_consistency():
    """检查数据一致性"""
    print("\n" + "="*50)
    print("检查数据文件...")
    print("="*50)
    
    import os
    import scanpy as sc
    
    # 检查数据路径
    data_path = "/home/wangluffy/projects/GNTD/GNTD_1/GNTD"
    
    # 查找表达数据文件
    expr_files = []
    for file in os.listdir(data_path):
        if file.endswith(('.mtx', '.csv', '.tsv', '.txt', '.h5ad')):
            expr_files.append(file)
            print(f"找到数据文件: {file}")
    
    if not expr_files:
        print("未找到数据文件")
        return False
    
    # 尝试读取第一个找到的文件
    test_file = os.path.join(data_path, expr_files[0])
    try:
        if test_file.endswith('.h5ad'):
            adata = sc.read_h5ad(test_file)
        elif test_file.endswith('.csv'):
            adata = sc.read_csv(test_file)
        elif test_file.endswith('.mtx'):
            # 需要同时有genes和barcodes文件
            print("MTX格式需要额外的genes.tsv和barcodes.tsv文件")
            return True
        else:
            adata = sc.read_text(test_file)
        
        print(f"✓ 成功读取数据: {adata.shape}")
        
        # 检查数据质量
        print(f"数据形状: {adata.n_obs} 细胞 x {adata.n_vars} 基因")
        
        # 检查缺失值
        if hasattr(adata.X, 'data'):
            X_data = adata.X.data
        else:
            X_data = adata.X.flatten()
        
        nan_count = np.isnan(X_data).sum()
        inf_count = np.isinf(X_data).sum()
        
        print(f"NaN数量: {nan_count}")
        print(f"Inf数量: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("⚠️ 数据中包含NaN或Inf值")
            return False
        
        # 检查零值比例
        zero_count = (X_data == 0).sum()
        total_count = len(X_data)
        print(f"零值比例: {zero_count/total_count*100:.2f}%")
        
        # 检查细胞总和
        if hasattr(adata.X, 'sum'):
            cell_sums = adata.X.sum(axis=1)
        else:
            cell_sums = adata.X.sum(axis=1)
        
        zero_cells = (cell_sums == 0).sum()
        print(f"零表达细胞数: {zero_cells}")
        
        if zero_cells == adata.n_obs:
            print("✗ 所有细胞都没有表达！")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 读取数据失败: {e}")
        return False

def check_gntd_code():
    """检查GNTD代码中的数据预处理部分"""
    print("\n" + "="*50)
    print("检查GNTD预处理代码...")
    print("="*50)
    
    import inspect
    import preprocessing
    
    # 获取preprocessing函数的源代码
    try:
        source = inspect.getsource(preprocessing.preprocessing)
        
        # 检查关键步骤
        checks = {
            '数据标准化': 'sc.pp.normalize_total' in source,
            '对数转换': 'sc.pp.log1p' in source,
            '高变基因': 'sc.pp.highly_variable_genes' in source,
            'PCA': 'sc.pp.pca' in source,
            'NaN处理': 'nan_to_num' in source or 'np.isnan' in source
        }
        
        for step, present in checks.items():
            if present:
                print(f"✓ {step}: 已实现")
            else:
                print(f"⚠️ {step}: 未找到")
                
        return True
    except Exception as e:
        print(f"✗ 无法检查代码: {e}")
        return False

def main():
    print("\n" + "🔍 GNTD环境检测")
    print("="*50)
    
    results = []
    
    # 执行各项检查
    results.append(("Python包", check_python_packages()))
    results.append(("R和mclust", check_r_and_mclust()))
    results.append(("rpy2连接", check_rpy2()))
    results.append(("数据一致性", check_data_consistency()))
    results.append(("GNTD代码", check_gntd_code()))
    
    # 总结
    print("\n" + "="*50)
    print("检测总结")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有检测通过！环境配置正确。")
        print("\n如果仍然遇到NaN/Inf错误，问题可能在数据本身。")
        print("建议运行数据清理脚本。")
    else:
        print("\n⚠️ 部分检测未通过，请根据上述提示修复。")
        
        # 提供具体建议
        if not results[1][1]:  # R/mclust失败
            print("\n修复R/mclust:")
            print("1. 安装R: sudo apt-get install r-base")
            print("2. 安装mclust: 在R中运行 install.packages('mclust')")
        
        if not results[2][1]:  # rpy2失败
            print("\n修复rpy2:")
            print("pip install --upgrade rpy2")
            print("可能需要设置R_HOME环境变量")
        
        if not results[3][1]:  # 数据问题
            print("\n数据问题解决方案:")
            print("1. 检查数据文件是否完整")
            print("2. 运行数据清理脚本")
            print("3. 确认数据格式正确")

if __name__ == "__main__":
    main()