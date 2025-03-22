import numpy as np

# 示例 self.action 数组
action = np.array([0.004, np.nan, 0.004, np.nan])

# 使用 numpy.isnan() 检查并替换 NaN 值为 0
action[np.isnan(action)] = 0

# 打印处理后的 self.action
print(action)
