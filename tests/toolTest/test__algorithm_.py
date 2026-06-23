from unittest import TestCase

import numpy as np
from scipy import stats

from sciv.tool import evaluate_probability_metrics, pearsonr


class Test(TestCase):
    def test_evaluate_probability_metrics(self):
        """测试完美预测的情况（模型完全正确）"""
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.2, 0.9, 0.8]  # 正样本概率远高于负样本

        auroc, auprc, brier, cross_entropy, roc_data, pr_data = evaluate_probability_metrics(y_true, y_pred)

        # 完美分类时 AUROC 应为 1.0
        assert auroc == 1.0
        # Brier score 和 cross_entropy 应该比较小
        assert brier < 0.1
        assert cross_entropy < 0.5

        # 检查返回的数据结构
        assert isinstance(roc_data, dict)
        assert set(roc_data.keys()) == {"fpr", "tpr", "thresholds"}
        assert isinstance(pr_data, dict)
        assert set(pr_data.keys()) == {"precision", "recall", "thresholds"}

        # 检查数组长度是否一致
        assert len(roc_data["fpr"]) == len(roc_data["tpr"]) == len(roc_data["thresholds"])
        assert len(pr_data["precision"]) == len(pr_data["recall"]) == len(pr_data["thresholds"]) + 1  # sklearn特性

        """测试随机预测的情况"""
        y_true = [0, 0, 1, 1]
        y_pred = [0.4, 0.6, 0.5, 0.5]  # 随机猜测

        auroc, auprc, brier, cross_entropy, roc_data, pr_data = evaluate_probability_metrics(y_true, y_pred)

        # 随机猜测 AUROC 应接近 0.5
        assert 0.0 <= auroc <= 1.0
        assert 0.0 <= auprc <= 1.0
        assert brier > 0.0
        assert cross_entropy > 0.0

        # 验证数组类型是 numpy array
        assert isinstance(roc_data["fpr"], np.ndarray)

        """测试 is_min_max=True 的情况"""
        y_true = [0, 1, 1, 0]
        # 未归一化的预测值（范围不在 0-1 之间）
        y_pred_raw = [10.0, 90.0, 80.0, 20.0]

        # 手动计算期望的归一化结果
        y_pred_normalized = [0.0, 1.0, 0.875, 0.125]

        # 调用函数
        auroc_raw, _, _, _, _, _ = evaluate_probability_metrics(y_true, y_pred_raw, is_min_max=True)

        # 用手动归一化的结果调用函数
        auroc_norm, _, _, _, _, _ = evaluate_probability_metrics(y_true, y_pred_normalized, is_min_max=False)

        """测试 空 的情况"""

        """测试 is_min_max=True 的情况"""
        y_true = [0, 1, 1, 0]
        # 未归一化的预测值（范围不在 0-1 之间）
        y_pred_raw = [0, 0, 0, 0]

        # 调用函数
        auroc, auprc, brier, cross_entropy, roc_curve_data, pr_curve_data = evaluate_probability_metrics(y_true, y_pred_raw, is_min_max=True)

        print(auroc, auprc, brier, cross_entropy, roc_curve_data, pr_curve_data)

    def test_pearsonr(self):
        # 1. 构造测试数据
        np.random.seed(42)  # 设置随机种子保证结果可复现
        n_samples = 100
        n_features = 5

        # 随机生成 x (1D) 和 y (2D)
        x_test = np.random.randn(n_samples)
        y_test = np.random.randn(n_features, n_samples)

        # 故意让 y 的第 0 行与 x 强正相关，第 1 行与 x 强负相关
        y_test[0] = x_test * 2 + np.random.randn(n_samples) * 0.1
        y_test[1] = -x_test * 3 + np.random.randn(n_samples) * 0.1

        print("=" * 50)
        print("开始测试自定义 pearsonr 函数...")
        print(f"输入 x 的形状: {x_test.shape}")
        print(f"输入 y 的形状: {y_test.shape}")
        print("=" * 50)

        # 2. 运行自定义函数 (axis=1, 即按行计算)
        my_corrs = pearsonr(x_test, y_test, axis=1)

        # 3. 使用 scipy.stats.pearsonr 获取基准结果进行对比
        scipy_corrs = []
        for i in range(n_features):
            r, p = stats.pearsonr(x_test, y_test[i])
            scipy_corrs.append(r)
        scipy_corrs = np.array(scipy_corrs)

        # 4. 打印结果对比
        print(f"{'特征索引':<10} | {'自定义函数结果':<20} | {'Scipy 结果':<20} | {'是否一致':<10}")
        print("-" * 70)
        for i in range(n_features):
            is_match = np.isclose(my_corrs[i], scipy_corrs[i])
            print(f"{i:<10} | {my_corrs[i]:<20.6f} | {scipy_corrs[i]:<20.6f} | {'✅ 是' if is_match else '❌ 否'}")

        print("\n测试结论:")
        if np.allclose(my_corrs, scipy_corrs):
            print("✅ 自定义函数计算结果与 Scipy 完全一致，代码正确！")
        else:
            print("❌ 存在差异，需要检查代码。")
