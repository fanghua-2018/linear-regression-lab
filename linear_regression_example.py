"""简单的一元线性回归示例（Python）。

运行方式：
    python linear_regression_example.py
"""

from __future__ import annotations


def fit_linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    """使用最小二乘法拟合 y = w*x + b，返回 (w, b)。"""
    if len(x) != len(y):
        raise ValueError("x 与 y 的长度必须一致")
    if len(x) < 2:
        raise ValueError("至少需要两个样本点")

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    if denominator == 0:
        raise ValueError("x 的方差为 0，无法拟合直线")

    w = numerator / denominator
    b = y_mean - w * x_mean
    return w, b


def predict(x: float, w: float, b: float) -> float:
    return w * x + b


if __name__ == "__main__":
    # 构造一组简单数据：y 约等于 2x + 1
    train_x = [1, 2, 3, 4, 5]
    train_y = [3.1, 5.0, 7.2, 9.1, 11.0]

    w, b = fit_linear_regression(train_x, train_y)
    print(f"拟合结果: y = {w:.3f}x + {b:.3f}")

    x_new = 6
    y_pred = predict(x_new, w, b)
    print(f"当 x={x_new} 时，预测 y={y_pred:.3f}")
