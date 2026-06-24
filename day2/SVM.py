import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# ============================================================
# 【修复1】解决中文显示问题
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # Windows优先用黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ============================================================
# 第一步：构造一个线性可分的二维数据集
# ============================================================
np.random.seed(42)

X_pos = np.random.randn(30, 2) * 0.5 + [3, 3]
X_neg = np.random.randn(30, 2) * 0.5 + [-3, -3]

X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(30), -np.ones(30)])

# ============================================================
# 第二步：训练硬间隔 SVM
# ============================================================
svm = SVC(kernel='linear', C=np.inf)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]
support_vectors = svm.support_vectors_

print(f"w = {w}")
print(f"b = {b:.4f}")
print(f"支持向量个数: {len(support_vectors)}")

# ============================================================
# 第三步：可视化
# ============================================================
plt.figure(figsize=(9, 7))

# 绘制样本点
plt.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', marker='o',
            label='正类 (+1)', edgecolors='k', s=60)
plt.scatter(X_neg[:, 0], X_neg[:, 1], c='red', marker='s',
            label='负类 (-1)', edgecolors='k', s=60)

# 绘制支持向量
plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
            facecolors='none', edgecolors='green', linewidths=2,
            s=150, label=f'支持向量 ({len(support_vectors)})')

# 创建网格并计算决策函数值
xx, yy = np.meshgrid(np.linspace(-6, 6, 300),
                      np.linspace(-6, 6, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = svm.decision_function(grid_points).reshape(xx.shape)

# 绘制决策边界和间隔线
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
plt.contour(xx, yy, Z, levels=[1], colors='blue', linestyles='--', linewidths=1.5)
plt.contour(xx, yy, Z, levels=[-1], colors='red', linestyles='--', linewidths=1.5)
plt.contourf(xx, yy, Z, levels=[-1, 1], colors='yellow', alpha=0.15)

plt.title('硬间隔 SVM (C=∞)', fontsize=16)
plt.xlabel('x₁', fontsize=13)
plt.ylabel('x₂', fontsize=13)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# ============================================================
# 【修复2】绕过 PyCharm 后端 Bug
# ============================================================
# 方法A（推荐）：保存图片到文件，不依赖 PyCharm 交互式窗口
plt.savefig('hard_margin_svm.png', dpi=150, bbox_inches='tight')
print("图片已保存为 hard_margin_svm.png")

# 方法B：如果你仍想弹窗显示，取消下面注释（需先升级PyCharm）
# plt.show()