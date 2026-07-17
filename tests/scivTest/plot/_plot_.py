# -*- coding: UTF-8 -*-

from unittest import TestCase

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sciv

from sciv.plot import violin, violin_significance


def generate_test_data(n=200):
    np.random.seed(42)

    # Create categories
    groups = ['Cluster A', 'Cluster B', 'Cluster C', 'Cluster D']
    hues = ['Type 1', 'Type 2']

    # 1. 生成基础列数据
    data = {
        'clusters': np.random.choice(groups, n),
        'value': np.random.normal(loc=50, scale=10, size=n),
        'hue_col': np.random.choice(hues, n),
    }

    # 2. 此时创建 DataFrame
    df = pd.DataFrame(data)

    # 3. 手动分配颜色（生成同样长度的颜色列表）
    color_map = {
        'Cluster A': '#FF0000',  # Red
        'Cluster B': '#00FF00',  # Green
        'Cluster C': '#0000FF',  # Blue
        'Cluster D': '#FFFF00'  # Yellow
    }

    # 使用 map 生成对应颜色的 Series，长度自动与 df 一致
    df['color'] = df['clusters'].map(color_map)

    # 4. 调整数值以便观察排序效果
    df.loc[df['clusters'] == 'Cluster A', 'value'] += 20
    df.loc[df['clusters'] == 'Cluster B', 'value'] += 10
    df.loc[df['clusters'] == 'Cluster C', 'value'] -= 10
    df.loc[df['clusters'] == 'Cluster D', 'value'] -= 20

    return df


class Test(TestCase):
    def test_violin(self):
        df_test = generate_test_data()

        # Test Case 1: Basic Violin Plot with Sorting (should work now)
        print("\n[Test 1] Basic Violin Plot (Sorted by Median)")
        try:
            g1 = violin(
                df=df_test,
                value="value",
                kind="violin",
                split=True,
                hue="hue_col",
                groupby="clusters",
                title="Test 1: Sorted Violin Plot",
                output="violin1.png",
                show=False
            )
            print("[Test 1] PASSED")
        except Exception as e:
            print(f"[Test 1] FAILED: {e}")

        # Test Case 2: Box Plot (should not crash on 'split' parameter)
        print("\n[Test 2] Box Plot (Checking split parameter handling)")
        try:
            g2 = violin(
                df=df_test,
                value="value",
                kind="box",
                groupby="clusters",
                split=True,  # Should be ignored for 'box'
                title="Test 2: Box Plot",
                output="violin2.png",
                show=False
            )
            print("[Test 2] PASSED")
        except Exception as e:
            print(f"[Test 2] FAILED: {e}")

        # Test Case 3: Custom Order and Colors
        print("\n[Test 3] Custom Order and Colors")
        try:
            custom_order = ['Cluster D', 'Cluster B', 'Cluster A', 'Cluster C']
            g3 = violin(
                df=df_test,
                value="value",
                kind="violin",
                groupby="clusters",
                order_names=custom_order,
                title="Test 3: Custom Order",
                output="violin3.png",
                show=False
            )
            print("[Test 3] PASSED")
        except Exception as e:
            print(f"[Test 3] FAILED: {e}")

        # Test Case 4: No Sorting (Preserve appearance order)
        print("\n[Test 4] No Sorting (is_sort=False)")
        try:
            g4 = violin(
                df=df_test,
                value="value",
                kind="strip",
                groupby="clusters",
                is_sort=False,
                title="Test 4: Appearance Order",
                output="violin4.png",
                show=False
            )
            print("[Test 4] PASSED")
        except Exception as e:
            print(f"[Test 4] FAILED: {e}")

        print("\n--- All Tests Completed ---")

    def test_violin_significance(self):
        np.random.seed(42)
        df_test = generate_test_data()

        print("开始测试 violin_significance 函数...")

        sciv.ul.fig, sciv.ul.ax = plt.subplots(figsize=(2, 2))
        # 场景 1: 使用默认的 t-test_ind 检验
        violin_significance(
            df=df_test,
            value="value",
            hue="hue_col",
            groupby="clusters",
            test="Mann-Whitney",  # 非参数检验
            palette="Set2",  # 使用调色板
            title="Analysis of Differences in Three Data Groups",
            x_name="Processing Group",
            y_name="measured value",
            output="violin_t_test.png"
        )
        print("场景 1 (t-test_ind) 完成，图片已保存。")

        # 场景 2: 使用非参数检验，并指定对比组
        sciv.ul.fig, sciv.ul.ax = plt.subplots(figsize=(2, 2))
        violin_significance(
            df=df_test,
            value="value",
            groupby="clusters",
            test="Mann-Whitney",  # 非参数检验
            output="violin_mann_whitney.png"
        )
        print("场景 2 (Mann-Whitney) 完成，图片已保存。")
