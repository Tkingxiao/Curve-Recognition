import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.special import comb, erf, gamma
from scipy.interpolate import lagrange, CubicSpline
import random
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class AdvancedCurveGenerator:
    def __init__(self, output_dir="advanced_curves"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_x_range(self, large_range=False):
        """生成随机的x范围 - 支持更大范围"""
        if large_range:
            start = random.uniform(-50, -20)
            length = random.uniform(30, 100)
        else:
            start = random.uniform(-10, 0)
            length = random.uniform(2, 20)

        end = start + length
        num_points = random.randint(200, 1000)  # 增加点数
        return np.linspace(start, end, num_points)

    # ============ 基础函数（扩展范围）============
    def basic_function(self, func_type, large_range=False):
        """生成基本函数"""
        x = self.generate_x_range(large_range)

        # 更大的随机参数范围
        a = random.uniform(0.1, 10.0)
        b = random.uniform(-10.0, 10.0)
        c = random.uniform(0.1, 5.0)
        d = random.uniform(-5.0, 5.0)
        e = random.uniform(-2.0, 2.0)

        if func_type == "sin":
            y = a * np.sin(c * x + d) + b
        elif func_type == "cos":
            y = a * np.cos(c * x + d) + b
        elif func_type == "tan":
            # 避免奇点
            x_safe = x[np.abs(np.cos(c * x + d)) > 0.05]  # 更宽松的条件
            if len(x_safe) < 10:
                x_safe = np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 200)
            y = a * np.tan(c * x_safe + d) + b
            return x_safe, y
        elif func_type == "exp":
            y = a * np.exp(c * x) + b
        elif func_type == "log":
            # 确保x为正
            x_positive = x[x > 0.01]
            if len(x_positive) < 10:
                x_positive = np.linspace(0.1, 10, 200)
            y = a * np.log(c * x_positive + 0.1) + b
            return x_positive, y
        elif func_type == "polynomial":
            # 更高阶多项式
            degree = random.randint(2, 8)
            coeffs = [random.uniform(-5, 5) for _ in range(degree + 1)]
            y = np.polyval(coeffs, x)
        elif func_type == "sqrt":
            # 平方根函数
            x_positive = x[x >= 0]
            if len(x_positive) < 10:
                x_positive = np.linspace(0, 10, 200)
            y = a * np.sqrt(c * x_positive + 0.1) + b
            return x_positive, y
        elif func_type == "abs":
            y = a * np.abs(c * x + d) + b
        elif func_type == "sigmoid":
            y = a / (1 + np.exp(-c * (x - d))) + b
        elif func_type == "gaussian":
            y = a * np.exp(-((x - b) ** 2) / (2 * c**2)) + d
        else:
            y = x

        return x, y

    # ============ 特殊函数 ============
    def special_function(self, func_type):
        """生成特殊函数曲线"""
        x = self.generate_x_range(large_range=True)

        if func_type == "bessel":
            # 贝塞尔函数（近似）
            from scipy.special import jv

            order = random.randint(0, 3)
            y = jv(order, x)
        elif func_type == "error":
            # 误差函数
            y = erf(x)
        elif func_type == "gamma_func":
            # Gamma函数（需要正数）
            x_positive = x[x > 0.1]
            y = gamma(x_positive)
            return x_positive, y
        elif func_type == "sinc":
            y = np.sinc(x / np.pi)
        elif func_type == "airy":
            # 艾里函数近似
            from scipy.special import airy

            Ai, Aip, Bi, Bip = airy(x)
            y = Ai  # 艾里Ai函数
        elif func_type == "legendre":
            # 勒让德多项式近似
            from numpy.polynomial.legendre import Legendre

            coeffs = [random.uniform(-2, 2) for _ in range(random.randint(3, 6))]
            poly = Legendre(coeffs)
            y = poly(x)
        elif func_type == "chebyshev":
            # 切比雪夫多项式
            from numpy.polynomial.chebyshev import Chebyshev

            coeffs = [random.uniform(-2, 2) for _ in range(random.randint(3, 6))]
            poly = Chebyshev(coeffs)
            y = poly(x)
        elif func_type == "fourier_series":
            # 傅里叶级数
            y = np.zeros_like(x)
            n_terms = random.randint(3, 8)
            for n in range(1, n_terms + 1):
                a_n = random.uniform(-1, 1) / n
                b_n = random.uniform(-1, 1) / n
                y += a_n * np.sin(n * x) + b_n * np.cos(n * x)
        else:
            y = np.sin(x)

        return x, y

    # ============ 贝塞尔曲线（改进）============
    def bezier_curve(self, num_points=200):
        """生成随机贝塞尔曲线"""
        n_ctrl = random.randint(3, 10)  # 更多控制点
        ctrl_points = []

        # 生成更大范围的控制点
        for i in range(n_ctrl):
            x = i * random.uniform(0.5, 3.0) + random.uniform(-2, 2)
            y = random.uniform(-20, 20)
            ctrl_points.append([x, y])

        ctrl_points = np.array(ctrl_points)

        # 使用de Casteljau算法
        def de_casteljau(t, points):
            if len(points) == 1:
                return points[0]
            new_points = []
            for i in range(len(points) - 1):
                x = (1 - t) * points[i][0] + t * points[i + 1][0]
                y = (1 - t) * points[i][1] + t * points[i + 1][1]
                new_points.append([x, y])
            return de_casteljau(t, new_points)

        t = np.linspace(0, 1, num_points)
        curve = np.array([de_casteljau(ti, ctrl_points) for ti in t])

        return curve[:, 0], curve[:, 1]

    # ============ 材料应力-应变曲线（多种类型）============
    def stress_strain_curve(self, material_type=None):
        """生成各种材料的应力-应变曲线"""
        if material_type is None:
            material_type = random.choice(
                [
                    "steel",
                    "aluminum",
                    "rubber",
                    "composite",
                    "ceramic",
                    "polymer",
                    "biological",
                    "superelastic",
                ]
            )

        if material_type == "steel":
            # 钢材 - 明显的屈服平台
            strain = np.linspace(0, random.uniform(0.15, 0.25), 300)
            E = random.uniform(180, 220)  # GPa
            sigma_y = random.uniform(200, 400)  # MPa
            strain_y = sigma_y / E

            stress = np.zeros_like(strain)
            elastic = strain <= strain_y
            plastic = ~elastic

            stress[elastic] = E * strain[elastic]

            # 屈服平台
            plateau_length = random.uniform(0.005, 0.015)
            plateau_mask = plastic & (strain <= strain_y + plateau_length)
            stress[plateau_mask] = sigma_y

            # 应变硬化
            hardening_mask = plastic & (strain > strain_y + plateau_length)
            strain_hardening = strain[hardening_mask] - strain_y - plateau_length
            stress[hardening_mask] = (
                sigma_y + random.uniform(0.5, 2.0) * E * strain_hardening
            )

        elif material_type == "aluminum":
            # 铝合金 - 连续屈服
            strain = np.linspace(0, random.uniform(0.08, 0.15), 300)
            E = random.uniform(60, 80)
            sigma_0 = random.uniform(100, 300)
            n = random.uniform(0.1, 0.3)  # 硬化指数

            stress = sigma_0 * (1 + E / sigma_0 * strain) ** n

        elif material_type == "rubber":
            # 橡胶 - 超弹性
            strain = np.linspace(0, random.uniform(1.0, 3.0), 400)  # 大应变
            C1 = random.uniform(0.5, 2.0)
            C2 = random.uniform(0.1, 0.5)

            # Mooney-Rivlin模型
            stress = 2 * (C1 + C2 / strain) * (strain - 1 / strain**2)

        elif material_type == "composite":
            # 复合材料 - 脆性断裂
            strain = np.linspace(0, random.uniform(0.02, 0.04), 200)
            E = random.uniform(100, 300)
            sigma_ult = random.uniform(300, 800)

            stress = E * strain
            # 线性直到断裂
            stress[strain > sigma_ult / E] = 0

        elif material_type == "ceramic":
            # 陶瓷 - 完全脆性
            strain = np.linspace(0, random.uniform(0.001, 0.003), 150)
            E = random.uniform(200, 400)
            sigma_ult = random.uniform(100, 400)

            stress = E * strain
            stress[strain > sigma_ult / E] = 0

        elif material_type == "polymer":
            # 聚合物 - 粘弹性
            strain = np.linspace(0, random.uniform(0.5, 1.5), 300)
            E1 = random.uniform(1, 5)  # 弹性模量
            E2 = random.uniform(0.1, 1)  # 粘性模量
            eta = random.uniform(0.5, 2)  # 粘度

            t = strain * 10  # 假设时间比例
            stress = E1 * strain + E2 * (1 - np.exp(-t / eta))

        elif material_type == "biological":
            # 生物材料 - J形曲线
            strain = np.linspace(0, random.uniform(0.5, 1.0), 300)
            a = random.uniform(0.1, 0.5)
            b = random.uniform(2, 5)

            stress = a * (np.exp(b * strain) - 1)

        elif material_type == "superelastic":
            # 超弹性（形状记忆合金）
            strain = np.linspace(0, random.uniform(0.06, 0.10), 400)

            # 相变平台
            sigma_start = random.uniform(200, 400)
            sigma_end = random.uniform(400, 600)
            plateau_strain = random.uniform(0.02, 0.04)

            stress = np.zeros_like(strain)

            # 初始弹性
            E_initial = random.uniform(50, 100)
            elastic_limit = sigma_start / E_initial
            elastic = strain <= elastic_limit
            stress[elastic] = E_initial * strain[elastic]

            # 相变平台
            plateau = (~elastic) & (strain <= elastic_limit + plateau_strain)
            stress[plateau] = (
                sigma_start
                + (sigma_end - sigma_start)
                * (strain[plateau] - elastic_limit)
                / plateau_strain
            )

            # 再弹性
            remaining = strain > elastic_limit + plateau_strain
            E_final = random.uniform(100, 200)
            stress[remaining] = sigma_end + E_final * (
                strain[remaining] - elastic_limit - plateau_strain
            )

        # 添加噪声
        noise_level = random.uniform(0.01, 0.05)
        stress += np.random.normal(0, noise_level * np.max(stress), len(stress))

        return strain, stress, material_type

    # ============ 物理/工程曲线 ============
    def physical_curve(self, curve_type):
        """生成物理/工程曲线"""
        x = self.generate_x_range(large_range=True)

        if curve_type == "damped_harmonic":
            # 阻尼简谐振动
            A = random.uniform(1, 10)
            omega = random.uniform(0.5, 5)
            phi = random.uniform(0, 2 * np.pi)
            beta = random.uniform(0.1, 1.0)

            y = A * np.exp(-beta * x) * np.cos(omega * x + phi)

        elif curve_type == "forced_vibration":
            # 受迫振动
            A = random.uniform(1, 5)
            omega_d = random.uniform(1, 3)  # 驱动频率
            omega_n = random.uniform(0.8, 1.2)  # 固有频率
            phi = random.uniform(0, np.pi / 2)
            damping = random.uniform(0.05, 0.2)

            # 简化的受迫振动响应
            y = (
                A
                / np.sqrt(
                    (omega_n**2 - omega_d**2) ** 2
                    + (2 * damping * omega_n * omega_d) ** 2
                )
                * np.sin(omega_d * x - phi)
            )

        elif curve_type == "heat_transfer":
            # 热传导温度分布
            T0 = random.uniform(100, 500)
            T_inf = random.uniform(20, 30)
            alpha = random.uniform(0.1, 1.0)

            y = T_inf + (T0 - T_inf) * np.exp(-alpha * x)

        elif curve_type == "fluid_flow":
            # 流体流动速度分布
            u_max = random.uniform(10, 50)
            R = random.uniform(0.1, 1.0)  # 管道半径
            r = np.abs(x)  # 假设x是径向位置

            # 泊肃叶流动
            y = u_max * (1 - (r / R) ** 2)
            y[r > R] = 0

        elif curve_type == "beam_deflection":
            # 梁的挠度曲线
            L = random.uniform(5, 20)  # 梁长度
            x_norm = x / L

            # 简支梁受均布载荷
            q = random.uniform(1, 10)
            EI = random.uniform(1000, 10000)

            y = q * L**4 / (24 * EI) * (x_norm**4 - 2 * x_norm**3 + x_norm)

        elif curve_type == "probability_dist":
            # 概率分布
            dist_type = random.choice(["normal", "lognormal", "weibull", "exponential"])

            if dist_type == "normal":
                mu = random.uniform(-2, 2)
                sigma = random.uniform(0.5, 2)
                y = (
                    1
                    / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                )

            elif dist_type == "exponential":
                lam = random.uniform(0.5, 2)
                x_positive = x[x >= 0]
                y = lam * np.exp(-lam * x_positive)
                return x_positive, y

        elif curve_type == "growth_curve":
            # 生长曲线（Logistic）
            L = random.uniform(10, 100)  # 上限
            k = random.uniform(0.1, 1.0)  # 增长率
            x0 = random.uniform(2, 8)  # 中点

            y = L / (1 + np.exp(-k * (x - x0)))

        return x, y

    # ============ 经济/金融曲线 ============
    def economic_curve(self):
        """生成经济/金融曲线"""
        x = self.generate_x_range(large_range=False)
        n = len(x)

        # 随机游走（股票价格）
        returns = np.random.normal(0.001, 0.02, n)  # 日收益率
        price = 100 * np.exp(np.cumsum(returns))

        # 添加趋势和周期
        trend = 0.0005 * np.arange(n)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 50)

        y = price * (1 + trend) + seasonal

        return x[:n], y

    # ============ 分形/混沌曲线 ============
    def fractal_curve(self):
        """生成分形曲线"""
        n_points = 500
        x = np.linspace(0, 10, n_points)

        # 随机分形布朗运动
        H = random.uniform(0.3, 0.7)  # 赫斯特指数
        fbm = np.zeros(n_points)

        for i in range(1, n_points):
            fbm[i] = fbm[i - 1] + np.random.randn() * (i ** (-H))

        # 归一化并缩放
        y = 10 * (fbm - np.min(fbm)) / (np.max(fbm) - np.min(fbm))
        y = y - np.mean(y)  # 居中

        return x, y

    # ============ 组合曲线 ============
    def composite_curve(self):
        """生成复杂组合曲线"""
        x = self.generate_x_range(large_range=True)
        y = np.zeros_like(x)

        # 组合3-6个不同函数
        n_components = random.randint(3, 6)
        components = random.sample(
            [
                ("sin", 1),
                ("cos", 1),
                ("poly", 2),
                ("exp", 1),
                ("log", 0.5),
                ("sigmoid", 1.5),
                ("gaussian", 0.8),
            ],
            n_components,
        )

        for func_type, weight in components:
            a = random.uniform(0.5, 3) * weight
            b = random.uniform(-5, 5)
            c = random.uniform(0.5, 2)
            d = random.uniform(-3, 3)

            if func_type == "sin":
                y += a * np.sin(c * x + d) + b
            elif func_type == "cos":
                y += a * np.cos(c * x + d) + b
            elif func_type == "poly":
                # 随机多项式
                coeffs = [random.uniform(-1, 1) for _ in range(random.randint(2, 4))]
                y += a * np.polyval(coeffs, x / 10) + b
            elif func_type == "exp":
                y += a * np.exp(0.1 * c * x) + b
            elif func_type == "log":
                y += a * np.log(np.abs(c * x) + 1) + b
            elif func_type == "sigmoid":
                y += a / (1 + np.exp(-c * (x - d))) + b
            elif func_type == "gaussian":
                y += a * np.exp(-((x - b) ** 2) / (2 * c**2))

        return x, y

    # ============ 主生成函数 ============
    def generate_curve(self, curve_type=None):
        """生成指定类型的曲线"""
        if curve_type is None:
            # 更丰富的曲线类型选择
            curve_type = random.choice(
                [
                    # 基础函数
                    "basic_sin",
                    "basic_cos",
                    "basic_tan",
                    "basic_exp",
                    "basic_log",
                    "basic_sqrt",
                    "basic_abs",
                    "basic_sigmoid",
                    "basic_gaussian",
                    # 特殊函数
                    "special_bessel",
                    "special_error",
                    "special_gamma",
                    "special_sinc",
                    "special_airy",
                    "special_legendre",
                    "special_chebyshev",
                    "special_fourier",
                    # 工程曲线
                    "bezier",
                    "stress_strain",
                    # 物理曲线
                    "physical_damped",
                    "physical_forced",
                    "physical_heat",
                    "physical_fluid",
                    "physical_beam",
                    "physical_prob",
                    "physical_growth",
                    # 其他
                    "economic",
                    "fractal",
                    "composite",
                ]
            )

        extra_points = None
        material_type = None

        if curve_type.startswith("basic_"):
            func_type = curve_type[6:]
            x, y = self.basic_function(func_type, large_range=random.random() > 0.5)

        elif curve_type.startswith("special_"):
            func_type = curve_type[8:]
            x, y = self.special_function(func_type)

        elif curve_type == "bezier":
            x, y = self.bezier_curve()

        elif curve_type == "stress_strain":
            x, y, material_type = self.stress_strain_curve()

        elif curve_type.startswith("physical_"):
            func_type = curve_type[9:]
            x, y = self.physical_curve(func_type)

        elif curve_type == "economic":
            x, y = self.economic_curve()

        elif curve_type == "fractal":
            x, y = self.fractal_curve()

        elif curve_type == "composite":
            x, y = self.composite_curve()

        else:
            x, y = self.basic_function("sin", large_range=True)

        return x, y, curve_type, extra_points, material_type

    # ============ 绘图函数 ============
    def plot_curve(
        self, x, y, curve_type, extra_points=None, material_type=None, filename=None
    ):
        """绘制曲线并保存"""
        fig_width = random.uniform(8, 15)  # 更大的图像
        fig_height = random.uniform(5, 10)
        plt.figure(figsize=(fig_width, fig_height))

        # 随机颜色和样式
        color = (random.random(), random.random(), random.random())
        linewidth = random.uniform(1.5, 4.0)

        # 根据曲线类型选择样式
        if curve_type == "stress_strain":
            plt.plot(x, y, color="red", linewidth=2.5, alpha=0.8)
            plt.fill_between(x, 0, y, alpha=0.1, color="red")
        elif curve_type == "bezier":
            plt.plot(
                x,
                y,
                color=color,
                linewidth=linewidth,
                marker="o",
                markersize=2,
                alpha=0.7,
            )
        elif "fractal" in curve_type or "economic" in curve_type:
            plt.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
            plt.fill_between(x, y, alpha=0.2, color=color)
        else:
            plt.plot(x, y, color=color, linewidth=linewidth, alpha=0.8)

        # 添加额外点
        if extra_points is not None:
            x_points, y_points = extra_points
            plt.scatter(
                x_points,
                y_points,
                color="red",
                s=80,
                zorder=5,
                edgecolors="black",
                linewidth=1.5,
            )

        # 网格和样式
        if random.random() > 0.4:
            plt.grid(True, alpha=random.uniform(0.1, 0.4), linestyle="--")

        # 标题和标签
        titles = {
            "basic_sin": "扩展正弦函数",
            "basic_cos": "扩展余弦函数",
            "basic_exp": "指数函数（大范围）",
            "basic_log": "对数函数",
            "stress_strain": f'材料应力-应变曲线 ({material_type if material_type else "随机材料"})',
            "bezier": "高阶贝塞尔曲线",
            "physical_damped": "阻尼振动曲线",
            "physical_heat": "热传导温度分布",
            "economic": "金融时间序列",
            "fractal": "分形布朗运动",
            "composite": "复杂组合函数",
        }

        title = titles.get(curve_type, "高级函数曲线")
        if random.random() > 0.3:
            plt.title(
                f"{title}\n范围: [{min(x):.1f}, {max(x):.1f}]",
                fontsize=14,
                fontweight="bold",
            )

        # 坐标轴标签
        if curve_type == "stress_strain":
            plt.xlabel("应变 ε", fontsize=12)
            plt.ylabel("应力 σ (MPa)", fontsize=12)
        elif "physical" in curve_type:
            labels = [
                ("时间 t (s)", "位移 x (m)"),
                ("位置 x (m)", "温度 T (°C)"),
                ("径向位置 r (m)", "速度 u (m/s)"),
            ]
            xlabel, ylabel = random.choice(labels)
            plt.xlabel(xlabel, fontsize=11)
            plt.ylabel(ylabel, fontsize=11)
        elif curve_type == "economic":
            plt.xlabel("时间 (天)", fontsize=11)
            plt.ylabel("价格 ($)", fontsize=11)
        else:
            if random.random() > 0.3:
                plt.xlabel("X", fontsize=11)
            if random.random() > 0.3:
                plt.ylabel("f(X)", fontsize=11)

        # 坐标轴范围
        x_margin = (max(x) - min(x)) * random.uniform(0.05, 0.15)
        y_margin = (max(y) - min(y)) * random.uniform(0.05, 0.15)
        plt.xlim(min(x) - x_margin, max(x) + x_margin)
        plt.ylim(min(y) - y_margin, max(y) + y_margin)

        # 添加文本说明
        if random.random() > 0.6:
            info_text = f"点数: {len(x)}\n范围: {max(x)-min(x):.1f}"
            plt.text(
                0.02,
                0.98,
                info_text,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # 保存图像
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{curve_type}_{timestamp}.png"

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()

        # 更高的DPI
        dpi = random.choice([150, 200, 300])
        plt.savefig(
            filepath, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.close()

        return filepath

    def generate_batch(self, num_images=300, curve_types=None, output_subdir=None):
        """批量生成图像"""
        if output_subdir:
            output_dir = os.path.join(self.output_dir, output_subdir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            original_dir = self.output_dir
            self.output_dir = output_dir

        generated_files = []
        stats = {}

        for i in range(num_images):
            print(f"生成第 {i+1}/{num_images} 张图像...", end="\r")

            try:
                if curve_types:
                    curve_type = random.choice(curve_types)
                else:
                    curve_type = None

                x, y, curve_type, extra_points, material_type = self.generate_curve(
                    curve_type
                )

                # 统计
                if curve_type not in stats:
                    stats[curve_type] = 0
                stats[curve_type] += 1

                filename = f"{curve_type}_{i+1:04d}.png"
                if material_type:
                    filename = f"{curve_type}_{material_type}_{i+1:04d}.png"

                filepath = self.plot_curve(
                    x, y, curve_type, extra_points, material_type, filename
                )
                generated_files.append(filepath)

            except Exception as e:
                print(f"\n生成第 {i+1} 张图像时出错: {e}")
                continue

        if output_subdir:
            self.output_dir = original_dir

        print(f"\n{'='*60}")
        print(f"完成！共生成 {len(generated_files)} 张图像")
        print(
            f"保存在: {os.path.abspath(output_dir if output_subdir else self.output_dir)}"
        )
        print("\n生成统计:")
        for curve_type, count in sorted(stats.items()):
            print(f"  {curve_type}: {count} 张")

        return generated_files, stats


# ============ 使用示例 ============
if __name__ == "__main__":
    # 创建高级生成器
    generator = AdvancedCurveGenerator("advanced_curves_large")

    # 1. 生成所有类型（300张）
    print("开始生成高级曲线图像...")
    files, stats = generator.generate_batch(num_images=300)

    # 2. 只生成应力-应变曲线（50张）
    # files, stats = generator.generate_batch(
    #     num_images=50,
    #     curve_types=['stress_strain'],
    #     output_subdir="stress_strain_curves"
    # )

    # 3. 生成大范围函数（100张）
    # files, stats = generator.generate_batch(
    #     num_images=100,
    #     curve_types=['basic_sin', 'basic_cos', 'basic_exp', 'composite'],
    #     output_subdir="large_range_curves"
    # )

    # 4. 生成物理/工程曲线（80张）
    # physical_curves = [f'physical_{c}' for c in
    #                   ['damped', 'forced', 'heat', 'fluid', 'beam', 'prob', 'growth']]
    # files, stats = generator.generate_batch(
    #     num_images=80,
    #     curve_types=physical_curves,
    #     output_subdir="physical_curves"
    # )

    print(f"\n图像总大小: {sum(os.path.getsize(f) for f in files)/1024/1024:.1f} MB")

    # 显示前几张图像的缩略图
    try:
        from PIL import Image
        import matplotlib.pyplot as plt

        print("\n前5张图像预览:")
        fig, axes = plt.subplots(1, min(5, len(files)), figsize=(15, 3))
        if min(5, len(files)) == 1:
            axes = [axes]

        for i, (ax, file) in enumerate(zip(axes, files[:5])):
            img = Image.open(file)
            ax.imshow(img)
            ax.set_title(f"{i+1}. {os.path.basename(file)[:20]}...")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\n要预览图像，请安装PIL: pip install pillow")
