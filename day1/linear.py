import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置中文显示（如果你运行时报错可以去掉这两行）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# ============================================================
# 第1步：生成模拟数据
# ============================================================
np.random.seed(42)  # 固定随机种子，保证每次运行结果一样

# X: 房屋面积（特征），100个样本，范围在 50~200 平方米
X = 50 + 150 * np.random.rand(100, 1)

# y: 房屋价格（目标值）= 基础价 + 单价×面积 + 随机噪声
# 真实关系：y = 50 + 0.8 * X + noise
noise = np.random.randn(100, 1) * 20  # 噪声，模拟现实中的波动
y = 50 + 0.8 * X + noise

# ============================================================
# 第2步：训练线性回归模型
# ============================================================
model = LinearRegression()
model.fit(X, y)  # fit() 就是让模型去学习数据中的规律

# 获取模型学到的参数
w = model.coef_[0][0]      # 权重（斜率）：每增加1平米，价格增加多少
b = model.intercept_[0]    # 偏置（截距）：基础价格

print(f"模型学到的方程: y = {w:.4f} * X + {b:.4f}")
print(f"（真实方程: y = 0.8 * X + 50）")

# ============================================================
# 第3步：可视化结果
# ============================================================
plt.figure(figsize=(10, 6))

# 画散点图（原始数据）
plt.scatter(X, y, color='steelblue', alpha=0.6, label='真实数据点', s=40)

# 画回归线（模型的预测）
X_line = np.array([[50], [200]])  # 线的起点和终点
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=2.5,
         label=f'回归线: y = {w:.2f}x + {b:.2f}')

# 标注一些预测误差（残差）
# ✅ 修复后（用 .item() 取出数组中的单个数值）
for i in range(0, 100, 15):
    y_pred_i = model.predict(X[i].reshape(1, -1))
    # .item() 把 numpy 数组中的值取出来变成普通 Python 数字
    xi = X[i].item()           # 例如: 从 array([123.4]) 变成 123.4
    yi = y[i].item()           # 真实值
    ypi = y_pred_i.item()      # 预测值
    plt.plot([xi, xi], [yi, ypi], color='gray',
             linestyle='--', alpha=0.5, linewidth=1)

plt.xlabel('房屋面积 (㎡)', fontsize=13)
plt.ylabel('房屋价格 (万元)', fontsize=13)
plt.title('简单线性回归：房屋面积 vs 房价', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_simple_linear_regression.png', dpi=150)
plt.show()