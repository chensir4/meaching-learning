import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import make_blobs

# 尝试设置中文字体（兼容不同操作系统，若失败则回退到英文）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# ==========================================
# 1. 生成线性可分的数据集
# ==========================================
# 生成两个簇，cluster_std 较小以保证它们绝对线性可分（硬间隔的前提）
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=0.6)

# SVM 的标准数学形式要求标签为 -1 和 +1
y = np.where(y == 0, -1, 1)

# ==========================================
# 2. 构建硬间隔 SVM 的优化问题
# ==========================================
# 我们的变量是 vars = [w1, w2, b]

# 目标函数: 最小化 1/2 * ||w||^2
def objective_function(vars):
    w = vars[:2]
    return 0.5 * np.dot(w, w)

# 约束条件: 对于每一个样本 i，必须满足 y_i * (w^T * x_i + b) >= 1
# 转换为 scipy 需要的不等式形式: y_i * (w^T * x_i + b) - 1 >= 0
constraints = []
for i in range(len(X)):
    cons = {
        'type': 'ineq',  # 不等式约束 (>= 0)
        # 注意: lambda 中使用 i=i 是为了绑定当前循环的 i 值（Python 闭包陷阱）
        'fun': lambda vars, i=i: y[i] * (np.dot(vars[:2], X[i]) + vars[2]) - 1
    }
    constraints.append(cons)

# 初始猜测值 [w1, w2, b]
initial_guess = np.zeros(3)

# ==========================================
# 3. 求解优化问题
# ==========================================
print("开始求解硬间隔 SVM 优化问题...")
# 使用 SLSQP 方法求解带约束的非线性优化问题
result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)

w_opt = result.x[:2]
b_opt = result.x[2]

print(f"优化状态: {result.message}")
print(f"求得的法向量 w: [{w_opt[0]:.4f}, {w_opt[1]:.4f}]")
print(f"求得的偏置 b: {b_opt:.4f}")

# ==========================================
# 4. 找出支持向量 (Support Vectors)
# ==========================================
# 支持向量是那些恰好落在间隔边界上的点，即满足 y_i * (w^T * x_i + b) == 1 的点
# 由于计算机数值计算有精度误差，我们设置一个很小的容差 tol
tol = 1e-4
support_vector_indices = []
for i in range(len(X)):
    if abs(y[i] * (np.dot(w_opt, X[i]) + b_opt) - 1) < tol:
        support_vector_indices.append(i)

support_vector_indices = np.array(support_vector_indices)
print(f"找到支持向量的数量: {len(support_vector_indices)}")

# ==========================================
# 5. 可视化结果
# ==========================================
plt.figure(figsize=(10, 8))

# 绘制所有数据点
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class +1', alpha=0.6, s=50)
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='Class -1', alpha=0.6, s=50)

# 用黑圈高亮标出支持向量
if len(support_vector_indices) > 0:
    plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1],
                s=200, facecolors='none', edgecolors='black', linewidths=2, label='Support Vectors')

