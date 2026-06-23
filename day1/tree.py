import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第1步：生成“双月”形状的非线性分类数据
# ============================================================
# make_moons 会生成两个交错的半圆，非常适合测试非线性分类器
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# ============================================================
# 第2步：训练决策树模型
# ============================================================
# max_depth=3 表示树最多只能问 3 层问题（限制深度防止过拟合）
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# 计算准确率
accuracy = clf.score(X, y)
print(f"模型准确率: {accuracy:.2%}")

# ============================================================
# 第3步：可视化（左图：决策边界，右图：树结构）
# ============================================================
fig = plt.figure(figsize=(16, 6))

# --- 左图：决策边界（模型在二维空间里是怎么切分数据的） ---
ax1 = fig.add_subplot(121)

# 1. 生成密集的网格点
h = 0.02  # 网格步长
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 2. 对网格中的每一个点进行预测
Z = clf.predict(grid_points)
Z = Z.reshape(xx.shape)

# 3. 画出决策区域的背景色
ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
# 4. 画出真实的数据点
scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                      edgecolors='black', s=40)

ax1.set_xlabel('特征 X1', fontsize=12)
ax1.set_ylabel('特征 X2', fontsize=12)
ax1.set_title(f'决策树的决策边界 (准确率: {accuracy:.1%})', fontsize=14)

# --- 右图：树结构（模型内部的 if-else 逻辑） ---
ax2 = fig.add_subplot(122)
plot_tree(clf,
          feature_names=['特征X1', '特征X2'],
          class_names=['类别0(蓝)', '类别1(红)'],
          filled=True,         # 节点填充颜色
          rounded=True,        # 圆角边框
          fontsize=10,
          ax=ax2)
ax2.set_title('决策树的内部结构 (If-Else 规则)', fontsize=14)

plt.tight_layout()
plt.savefig('06_decision_tree_basics.png', dpi=150)
plt.show()