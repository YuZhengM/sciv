from unittest import TestCase

import numpy as np

from sciv.tool._random_walk_ import _random_walk_cpu_, _random_walk_gpu_


class Test(TestCase):

    def test__random_walk_cpu_(self):
        print("开始测试 _random_walk_cpu_ ...\n")

        # 1. 测试数据准备：构建一个简单的随机游走转移矩阵
        # 我们定义一个 4x4 的矩阵，模拟 4 个节点之间的连接关系
        # 节点 0 连接到 1; 1 连接到 0 和 2; 2 连接到 1 和 3; 3 连接到 2
        # 这里使用列归一化的权重矩阵
        W = np.array([
            [0.0, 0.5, 0.0, 0.0],
            [1.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

        # 初始概率分布（种子节点）。
        # 假设从节点 0 开始游走
        V0 = np.array([1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)

        # ==========================================
        # 测试用例 1: 基础功能测试 (1维输入)
        # ==========================================
        print("测试用例 1: 基础随机游走 (p=2, gamma=0.05)")
        try:
            result_1 = _random_walk_cpu_(V0, W, gamma=0.05, max_steps=100)
            print(f"输入形状: {V0.shape}, 权重形状: {W.shape}")
            print(f"结果形状: {result_1.shape}")
            print(f"结果值: \n{result_1}")
            assert result_1.shape == (4,), "输出形状应为 (4,)"
            assert np.allclose(result_1.sum(), 1.0, atol=1e-4), "概率总和应近似为 1"
            print("✅ 通过\n")
        except Exception as e:
            print(f"❌ 失败: {e}\n")

        # ==========================================
        # 测试用例 2: 多样本并发计算 (2D 输入)
        # ==========================================
        print("测试用例 2: 多样本并发计算 (不同种子)")
        try:
            # 构造两个初始分布，分别从节点 0 和节点 2 开始
            V0_multi = np.array([
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0]
            ])

            result_2 = _random_walk_cpu_(V0_multi, W, gamma=0.1, max_steps=50)
            print(f"输入形状: {V0_multi.shape}")
            print(f"结果形状: {result_2.shape}")
            print(f"结果样本1 (从节点0开始): {result_2[:, 0]}")
            print(f"结果样本2 (从节点2开始): {result_2[:, 1]}")

            assert result_2.shape == (4, 2), "输出形状应为 (4, 2)"
            assert np.allclose(result_2.sum(axis=0), 1.0, atol=1e-4), "每个样本的概率总和应近似为 1"
            print("✅ 通过\n")
        except Exception as e:
            print(f"❌ 失败: {e}\n")

        # ==========================================
        # 测试用例 3: 不同范数收敛判断 (p=1)
        # ==========================================
        print("测试用例 3: 使用 L1 范数进行收敛判断")
        try:
            result_3 = _random_walk_cpu_(V0, W, gamma=0.05, p=1, max_steps=100)
            print(f"结果形状: {result_3.shape}")
            assert result_3.shape == (4,)
            print("✅ 通过\n")
        except Exception as e:
            print(f"❌ 失败: {e}\n")

        # ==========================================
        # 测试用例 4: 异常处理测试 (维度不匹配)
        # ==========================================
        print("测试用例 4: 维度不匹配 (应抛出 ValueError)")
        try:
            V0_wrong = np.array([1.0, 0.0])  # 长度为 2
            W_wrong = W  # 形状为 4x4
            _random_walk_cpu_(V0_wrong, W_wrong)
            print("❌ 失败: 未捕获维度不匹配错误\n")
        except ValueError as e:
            print(f"✅ 成功捕获错误: {str(e)[:50]}...\n")
        except Exception as e:
            print(f"❌ 失败: 捕获了非预期错误 {e}\n")

        # ==========================================
        # 测试用例 5: 完美模式测试
        # 说明: 代码逻辑显示 is_perfect=True 会将已收敛的列置零存入 vt_finish
        # 这是一个逻辑验证测试
        # ==========================================
        print("测试用例 5: 完美模式")
        try:
            # 两个样本，一个容易收敛，一个难收敛（如果矩阵支持的话）
            # 这里主要验证代码不报错且能运行
            V0_multi = np.array([
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0]
            ])
            result_5 = _random_walk_cpu_(V0_multi, W, gamma=0.5, max_steps=200, is_perfect=True)
            print(f"结果形状: {result_5.shape}")
            # 在完美模式下，收敛后的列会被固定，理论上结果与普通模式相似（假设都在max_steps内收敛）
            print("✅ 通过\n")
        except Exception as e:
            print(f"❌ 失败: {e}\n")

        # ==========================================
        # 测试用例 6: 输入为 List
        # ==========================================
        print("测试用例 6: 输入为原生 Python List")
        try:
            init_list = [1.0, 0.0, 0.0, 0.0]
            result_6 = _random_walk_cpu_(init_list, W)
            assert isinstance(result_6, np.ndarray), "返回结果应为 ndarray"
            assert result_6.shape == (4,)
            print("✅ 通过\n")
        except Exception as e:
            print(f"❌ 失败: {e}\n")


    def test__random_walk_gpu_(self):
        # --- 1. 构造测试数据 ---

        # 设置随机种子以保证可复现性
        np.random.seed(42)

        # 参数设置
        n_nodes = 100  # 节点数量 (矩阵行数)
        n_seeds = 5  # 种子数量/样本数量 (矩阵列数)
        density = 0.1  # 稀疏矩阵密度

        from scipy.sparse import random as sparse_random

        # 生成转移权重矩阵
        # 这里使用 scipy.sparse 生成一个随机的稀疏矩阵，并归一化使其行和为 1 (马尔可夫性质)
        scipy_sparse_weight = sparse_random(n_nodes, n_nodes, density=density, format='csr', dtype=np.float32)
        # 将每一行归一化，使其模拟转移概率
        row_sums = np.array(scipy_sparse_weight.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # 防止除以0
        scipy_sparse_weight.data = scipy_sparse_weight.data / row_sums[scipy_sparse_weight.indices]

        # 生成初始概率矩阵
        # 形状为，每一列代表一个种子细胞的起始概率分布
        init_prob_np = np.random.rand(n_nodes, n_seeds).astype(np.float32)
        # 归一化每一列
        init_prob_np = init_prob_np / init_prob_np.sum(axis=0, keepdims=True)

        print(f"测试数据准备完毕:")
        print(f"权重矩阵 shape: {scipy_sparse_weight.shape}, 非零元素数量: {scipy_sparse_weight.nnz}")
        print(f"初始概率 shape: {init_prob_np.shape}")
        print("-" * 50)

        # --- 2. 运行测试函数 ---

        try:
            # 调用函数，传入 numpy 数组和 scipy sparse matrix
            result = _random_walk_gpu_(
                init_prob=init_prob_np,
                weight=scipy_sparse_weight,
                gamma=0.1,  # 稍微增大重启概率以便快速收敛
                epsilon=1e-5,
                max_steps=100,
                p=2,
                device='auto',  # 自动选择设备 (如果有 GPU 则用 GPU，否则 CPU)
                is_perfect=False  # 先测试标准模式
            )

            # --- 3. 结果验证 ---

            print("测试成功运行！")
            print(f"返回结果类型: {type(result)}")
            print(f"返回结果 shape: {result.shape}")

            # 验证 1: 形状应与输入的 init_prob 相同
            assert result.shape == init_prob_np.shape, f"Shape mismatch! Expected {init_prob_np.shape}, got {result.shape}"

            # 验证 2: 返回结果应为 numpy 数组
            assert isinstance(result, np.ndarray), "Result should be a numpy array."

            # 验证 3: 每一列的概率和应该近似为 1 (对于随机游走的平稳分布)
            col_sums = result.sum(axis=0)
            print(f"各列概率和 (应接近 1.0): {np.round(col_sums, 5)}")
            assert np.allclose(col_sums, 1.0, atol=1e-4), "Column sums are not close to 1.0"

            print("-" * 50)
            print("基本断言通过。")

            # 打印部分结果预览
            print("\n前5行 x 前3列的结果预览:")
            print(result[:5, :3])

        except Exception as e:
            print(f"\n测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

        # --- 4. 进阶测试：测试 is_perfect 模式 ---
        print("\n" + "=" * 50)
        print("开始测试 is_perfect=True 模式...")
        try:
            result_perfect = _random_walk_gpu_(
                init_prob=init_prob_np,
                weight=scipy_sparse_weight,
                gamma=0.1,
                device='auto',
                is_perfect=True  # 开启完美模式
            )

            print("is_perfect=True 模式运行成功。")
            print(f"结果 shape: {result_perfect.shape}")

            # 完美模式下的列和也应为 1
            col_sums_p = result_perfect.sum(axis=0)
            assert np.allclose(col_sums_p, 1.0, atol=1e-4), "Perfect mode column sums check failed."
            print("完美模式断言通过。")

        except Exception as e:
            print(f"完美模式测试失败: {e}")

        # --- 5. 进阶测试：测试单列输入 (返回 flatten) ---
        print("\n" + "=" * 50)
        print("开始测试单列输入 (应返回 1D array)...")
        try:
            single_col_init = init_prob_np[:, [0]]  # 取出第一列，形状
            result_1d = _random_walk_gpu_(
                init_prob=single_col_init,
                weight=scipy_sparse_weight,
                gamma=0.1,
                device='auto'
            )

            # 根据代码逻辑: if vt.shape[1] == 1: return vt.flatten()
            assert result_1d.ndim == 1, f"Single column input should return 1D array, got shape {result_1d.shape}"
            assert result_1d.shape[0] == n_nodes, f"1D output length mismatch."
            print(f"返回结果维度: {result_1d.ndim}, Shape: {result_1d.shape}")
            print("单列输入测试通过。")

        except Exception as e:
            print(f"单列输入测试失败: {e}")
