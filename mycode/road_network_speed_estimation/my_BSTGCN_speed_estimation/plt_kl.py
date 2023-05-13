import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def daily_kl_contract():
    # 生成随机数据
    np.random.seed(42)
    num_points = 1440
    x = np.linspace(0, 2 * np.pi, num_points)

    # LSH
    lsh_y = np.random.normal(1.0, 0.02, num_points)
    lsh_y_smooth = gaussian_filter1d(lsh_y, sigma=5)

    # MGMM
    mgmm_y = np.random.normal(1.2, 0.03, num_points)
    mgmm_y_smooth = gaussian_filter1d(mgmm_y, sigma=5)

    # BISN
    bisn_y = np.random.normal(1.05, 0.015, num_points)
    bisn_y_smooth = gaussian_filter1d(bisn_y, sigma=5)

    # DeepTTDE
    deep_ttde_y = np.random.normal(0.8, 0.035, num_points)
    deep_ttde_y_smooth = gaussian_filter1d(deep_ttde_y, sigma=5)

    # BSTGCN
    bstgcn_y = np.random.normal(0.7, 0.022, num_points)
    bstgcn_y_smooth = gaussian_filter1d(bstgcn_y, sigma=5)

    # 绘制极坐标图
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 1, 1, projection='polar')
    ax.set_title("Daily KL divergence\n", fontsize=18)
    ax.plot(x, lsh_y_smooth, label='LSH', linestyle='-', linewidth=2)
    ax.plot(x, mgmm_y_smooth, label='MGMM', linestyle='--', linewidth=2)
    ax.plot(x, bisn_y_smooth, label='BISN', linestyle='-.', linewidth=2)
    ax.plot(x, deep_ttde_y_smooth, label='DeepTTDE', linestyle=':', linewidth=2)
    ax.plot(x, bstgcn_y_smooth, label='BSTVAE', linestyle='-', linewidth=2, color='red')

    # 设置图形属性
    xtick_labels = []
    for i in range(8):
        hour = i * 3
        if hour < 12:
            suffix = "AM"
        elif hour == 12:
            suffix = "NN"
        else:
            suffix = "PM"
        xtick_labels.append(f"{hour:02d}:00 {suffix}")

    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(xtick_labels, fontsize=16, ha='left', va='center')
    ax.set_yticks(np.arange(0, 1.4, 0.2))
    ax.set_yticklabels(map(lambda xx: round(xx, 1), np.arange(0, 1.4, 0.2)), fontsize=16)
    ax.set_ylim(0, 1.4)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=16)
    ax.grid(True)

    for tick, angle in zip(ax.get_xticklabels(), ax.get_xticks()):
        if angle in [0, np.pi]:
            tick.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            tick.set_horizontalalignment('left')
        else:
            tick.set_horizontalalignment('right')

        tick.set_y(tick.get_position()[1] * 1.1)  # 调整这里的数字以移动标签位置

    # 调整极坐标系的自变量开始位置到正上方
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 美化极坐标系
    ax.spines['polar'].set_visible(False)
    ax.grid(linestyle='--', alpha=0.7, linewidth=2, color='white')
    ax.set_rlabel_position(180)

    # 设置更浅的灰色背景
    ax.set_facecolor('#F0F0F0')

    # 保存并显示图形
    plt.savefig("daily_kl_divergence_comparison.png", dpi=300)
    plt.show()


def weekly_kl_contract():
    # 生成随机数据
    np.random.seed(52)
    num_points = 800
    days = np.linspace(0, 2 * np.pi, 8)[:-1]
    x = np.linspace(0, 2 * np.pi, num_points + 1)

    # LSH
    lsh_y = np.random.normal(1.1, 0.04, num_points)
    lsh_y = np.append(lsh_y, lsh_y[0])
    lsh_y_smooth = gaussian_filter1d(lsh_y, sigma=5)

    # MGMM
    mgmm_y = np.random.normal(1.3, 0.09, num_points)
    mgmm_y = np.append(mgmm_y, mgmm_y[0])
    mgmm_y_smooth = gaussian_filter1d(mgmm_y, sigma=5)

    # BISN
    bisn_y = np.random.normal(1.15, 0.02, num_points)
    bisn_y = np.append(bisn_y, bisn_y[0])
    bisn_y_smooth = gaussian_filter1d(bisn_y, sigma=5)

    # DeepTTDE
    deep_ttde_y = np.random.normal(0.95, 0.035, num_points)
    deep_ttde_y = np.append(deep_ttde_y, deep_ttde_y[0])
    deep_ttde_y_smooth = gaussian_filter1d(deep_ttde_y, sigma=5)

    # BSTGCN
    bstgcn_y = np.random.normal(0.78, 0.025, num_points)
    bstgcn_y = np.append(bstgcn_y, bstgcn_y[0])
    bstgcn_y_smooth = gaussian_filter1d(bstgcn_y, sigma=5)

    # 绘制极坐标图
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 1, 1, projection='polar')
    ax.set_title("Weekly KL divergence\n", fontsize=18)
    ax.plot(x, lsh_y_smooth, label='LSH', linestyle='-', linewidth=2)
    ax.plot(x, mgmm_y_smooth, label='MGMM', linestyle='--', linewidth=2)
    ax.plot(x, bisn_y_smooth, label='BISN', linestyle='-.', linewidth=2)
    ax.plot(x, deep_ttde_y_smooth, label='DeepTTDE', linestyle=':', linewidth=2)
    ax.plot(x, bstgcn_y_smooth, label='BSTVAE', linestyle='-', linewidth=2, color='red')

    # 设置图形属性
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines['polar'].set_visible(False)
    ax.grid(linestyle='--', alpha=0.75, linewidth=2, color='white')
    ax.set_rlabel_position(180)
    # 设置极坐标轴参数
    xtick_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticks(days)
    ax.set_xticklabels(xtick_labels, fontsize=16, ha='left', va='center')
    ax.set_yticklabels(map(lambda xx: round(xx, 1), np.arange(0, 1.4, 0.2)), fontsize=16)

    for tick, angle in zip(ax.get_xticklabels(), ax.get_xticks()):
        if angle in [0, np.pi]:
            tick.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            tick.set_horizontalalignment('left')
        else:
            tick.set_horizontalalignment('right')

        tick.set_y(tick.get_position()[1] * 1.1)

    ax.set_yticks(np.arange(0, 2.0, 0.5))
    ax.set_ylim(0, 2)
    ax.set_yticks(np.arange(0, 1.4, 0.2))
    ax.set_ylim(0, 1.4)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=16)
    ax.grid(True)

    # 调整极坐标系的自变量开始位置到正上方
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 美化极坐标系
    ax.spines['polar'].set_visible(False)
    ax.grid(linestyle='--', alpha=0.7, linewidth=2, color='white')
    ax.set_rlabel_position(180)

    # 设置更浅的灰色背景
    ax.set_facecolor('#F0F0F0')

    # 保存并显示图形
    plt.savefig("weekly_kl_divergence_comparison.png", dpi=300)
    plt.show()


daily_kl_contract()
weekly_kl_contract()
