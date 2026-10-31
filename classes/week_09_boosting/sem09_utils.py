"""Рисунки семинара 9 «Градиентный бустинг».

Функции здесь только рисуют: модели обучают и эксперименты проводят
в ноутбуке, а сюда передают обученные модели и посчитанные числа.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

plt.rcParams.update({
    "figure.figsize": (8, 4.5),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})

BLUE, BLACK, OCHRE, GREY = "#0072CE", "#0F1418", "#C98A3C", "#9CA3AF"
LIGHT = "#EEF2F6"
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
POINT_COLORS = ListedColormap([BLUE, OCHRE])
PROBABILITY_COLORS = LinearSegmentedColormap.from_list("класс 0 — класс 1", [BLUE, "white", OCHRE])

LOWER_IS_BETTER = ("MSE", "ошибк", "время", "смещение", "разброс", "сумма")
NO_COLOR = ("число деревьев", "деревьев после", "дисперсия шума", "вес", "ответ")


# --- таблицы ------------------------------------------------------------------
def show_table(frame, digits=3, axis=0):
    """Таблица с заливкой: лучшее значение ярче; для ошибок и времени лучше меньшее."""
    numeric = list(frame.select_dtypes("number").columns)
    plain = [c for c in numeric if any(key in str(c) for key in NO_COLOR)]
    lower = [c for c in numeric if c not in plain and any(key in str(c) for key in LOWER_IS_BETTER)]
    higher = [c for c in numeric if c not in plain and c not in lower]

    styled = frame.style.format(precision=digits, subset=numeric)
    if higher:
        styled = styled.background_gradient(cmap="Greens", subset=higher, axis=axis)
    if lower:
        styled = styled.background_gradient(cmap="Greens_r", subset=lower, axis=axis)
    return styled


# --- деревья и схемы ------------------------------------------------------------
def tree_layout(structure):
    """Координаты узлов дерева: листья слева направо, внутренний узел — посередине между потомками."""
    positions = {}
    leaf_counter = [0]

    def place(node, depth):
        left_child = structure.children_left[node]
        right_child = structure.children_right[node]

        if left_child == -1:
            positions[node] = (leaf_counter[0], -depth)
            leaf_counter[0] += 1
            return

        place(left_child, depth + 1)
        place(right_child, depth + 1)
        middle_x = (positions[left_child][0] + positions[right_child][0]) / 2
        positions[node] = (middle_x, -depth)

    place(0, 0)
    return positions


def draw_tree(ax, model, detailed=True):
    """Схема дерева: вопросы в узлах и значения листьев; для большого дерева — только узлы и ребра.

    У дерева регрессии в листе — его значение, у классификатора — доля класса 1.
    """
    structure = model.tree_
    positions = tree_layout(structure)
    is_classifier = hasattr(model, "classes_")
    if structure.n_features == 1:
        feature_names = ["x"]
    else:
        feature_names = [f"x{j + 1}".translate(SUBSCRIPT) for j in range(structure.n_features)]

    for node, (x, y) in positions.items():
        left_child = structure.children_left[node]
        right_child = structure.children_right[node]
        if left_child == -1:
            continue
        for child, answer in [(left_child, "да"), (right_child, "нет")]:
            child_x, child_y = positions[child]
            ax.plot([x, child_x], [y, child_y], color=GREY, lw=1.2 if detailed else 0.6, zorder=1)
            if detailed:
                ax.text((x + child_x) / 2, (y + child_y) / 2, answer, ha="center", va="center", fontsize=8,
                        color=GREY, bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none"))

    for node, (x, y) in positions.items():
        is_leaf = structure.children_left[node] == -1

        if is_classifier:
            counts = structure.value[node, 0]
            leaf_value = counts[1] / counts.sum()
            leaf_text = f"p = {leaf_value:.2f}"
            is_positive = leaf_value > 0.5
        else:
            leaf_value = structure.value[node, 0, 0]
            leaf_text = f"{leaf_value:+.2f}"
            is_positive = leaf_value > 0

        if detailed and is_leaf:
            leaf_color = OCHRE if is_positive else BLUE
            ax.text(x, y, leaf_text, ha="center", va="center", fontsize=9, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=leaf_color, alpha=0.25, edgecolor=leaf_color))
        elif detailed:
            feature_name = feature_names[structure.feature[node]]
            question = f"{feature_name} ≤ {structure.threshold[node]:.2f}?"
            ax.text(x, y, question, ha="center", va="center", fontsize=9, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=BLACK))
        else:
            ax.scatter(x, y, s=10 if is_leaf else 14, color=OCHRE if is_leaf else BLACK, zorder=2)

    n_leaves = model.get_n_leaves()
    lowest_level = min(y for _, y in positions.values())
    ax.set_xlim(-0.8, n_leaves - 0.2)
    ax.set_ylim(lowest_level - 0.6, 0.6)
    ax.axis("off")


def box(ax, x, y, text, color="white", edge=BLACK, size=11):
    """Прямоугольник с текстом для схем."""
    ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=3,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor=edge))


def arrow(ax, start, end, color=GREY, bend=0.0):
    """Стрелка от start к end; bend — изгиб дуги."""
    ax.annotate("", xy=end, xytext=start, zorder=2,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, connectionstyle=f"arc3,rad={bend}"))


# --- раздел 1: смещение и разброс -----------------------------------------------
def plot_bias_variance_targets(seed=0):
    """Мишени: центр — истинный ответ, точки — предсказания моделей, обученных на разных выборках."""
    rng = np.random.default_rng(seed)
    cases = [
        ("малое смещение, малый разброс", "случайный лес, бустинг\nс подобранным числом деревьев", (0.0, 0.0), 0.05),
        ("малое смещение, большой разброс", "глубокое дерево,\nkNN с k = 1", (0.0, 0.0), 0.22),
        ("большое смещение, малый разброс", "линейная регрессия на синусе,\nпень", (0.05, 0.62), 0.05),
        ("большое смещение, большой разброс", "линейная регрессия по многим\nшумным признакам на малой выборке",
         (-0.15, 0.68), 0.22),
    ]
    rings = [(1.0, "white"), (0.75, "#9DBCD9"), (0.45, "white"), (0.15, "#B5524B")]

    fig, axes = plt.subplots(2, 2, figsize=(6.65, 7.2))
    for ax, (title, models, center, spread) in zip(axes.ravel(), cases):
        for radius, color in rings:
            ax.add_patch(plt.Circle((0, 0), radius, facecolor=color, edgecolor=BLACK, lw=1.0))
        points = rng.normal(center, spread, size=(20, 2))
        ax.scatter(points[:, 0], points[:, 1], s=16, color="#0B1F8A", zorder=3)
        ax.text(0, -1.06, models, ha="center", va="top", fontsize=8.5, style="italic", color=BLACK)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.45, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_tree_depths(models, test_mse, x_train, y_train, x_grid):
    """Сверху — устройство деревьев, снизу — их предсказания на синусе."""
    fig, axes = plt.subplots(2, len(models), figsize=(19, 9.5), gridspec_kw={"height_ratios": [1, 1.15]})

    for column, (name, model) in enumerate(models.items()):
        n_leaves = model.get_n_leaves()
        is_small = n_leaves <= 8

        ax_tree = axes[0, column]
        draw_tree(ax_tree, model, detailed=is_small)
        ax_tree.set_title(f"{name}\nглубина {model.get_depth()}, вопросов {n_leaves - 1}, листьев {n_leaves}", fontsize=13)

        ax_fit = axes[1, column]
        ax_fit.scatter(x_train[:, 0], y_train, s=14, color=GREY, label="обучающая выборка")
        ax_fit.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), color=OCHRE, lw=1.5, ls="--", label="sin x")
        ax_fit.plot(x_grid[:, 0], model.predict(x_grid), color=BLACK, lw=2, label="предсказание дерева")
        if is_small:
            thresholds = model.tree_.threshold[model.tree_.children_left != -1]
            for threshold in thresholds:
                ax_fit.axvline(threshold, color=BLUE, ls=":", lw=1)
        ax_fit.set_title(f"MSE на тестовом сете {test_mse[name]:.3f}", fontsize=12)
        ax_fit.set_xlabel("x")
        if column == 0:
            ax_fit.set_ylabel("y")
            ax_fit.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_fits(curves, x_train, y_train, x_grid, highlight=None, legend_outside=False, size=(10, 4.5)):
    """Обучающая выборка, sin x и кривые моделей; curves — {подпись: значения в точках x_grid}."""
    colors = [BLACK, BLUE, OCHRE, "#4DAF4A", GREY]
    plt.figure(figsize=size)
    plt.scatter(x_train[:, 0], y_train, s=14, color=GREY, label="обучающая выборка")
    if highlight is not None:
        plt.scatter(x_train[highlight, 0], y_train[highlight], s=60, facecolor="none", edgecolor="red", lw=1.5,
                    label="выбросы")
    plt.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), color=GREY, lw=1.5, ls="--", label="sin x")
    for (label, values), color in zip(curves.items(), colors):
        plt.plot(x_grid[:, 0], values, color=color, lw=2, label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    if legend_outside:
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    else:
        plt.legend(loc="lower left")
    plt.show()


def plot_fit_panels(panels, x_train, y_train, x_grid, size=(15, 4.6)):
    """Несколько графиков plot_fits рядом; panels — [(заголовок, {подпись: значения в точках x_grid})]."""
    colors = [BLACK, BLUE, OCHRE, "#4DAF4A", GREY]
    fig, axes = plt.subplots(1, len(panels), figsize=size, sharey=True)
    for ax, (title, curves) in zip(axes, panels):
        ax.scatter(x_train[:, 0], y_train, s=14, color=GREY, label="обучающая выборка")
        ax.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), color=GREY, lw=1.5, ls="--", label="sin x")
        for (label, values), color in zip(curves.items(), colors):
            ax.plot(x_grid[:, 0], values, color=color, lw=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("y")
    plt.tight_layout()
    plt.show()


def plot_ensemble_scheme():
    """Схемы трех ансамблей: бэггинг, стекинг, бустинг."""
    fig, (ax_bagging, ax_stacking, ax_boosting) = plt.subplots(3, 1, figsize=(18, 14),
                                                               gridspec_kw={"height_ratios": [1, 1, 1.25]})

    ax_bagging.set_xlim(0, 10)
    ax_bagging.set_ylim(0, 4)
    ax_bagging.axis("off")
    ax_bagging.set_title("бэггинг: модели одного типа обучают независимо, ответ — среднее", fontsize=14, loc="left")
    box(ax_bagging, 0.9, 2, "обучающая\nвыборка", color=LIGHT)
    for y, number in [(3.3, "1"), (2.0, "2"), (0.7, "B")]:
        box(ax_bagging, 3.2, y, f"бутстрэп-выборка {number}")
        box(ax_bagging, 5.9, y, f"глубокое дерево $b_{number}$", color=BLUE + "33", edge=BLUE)
        arrow(ax_bagging, (1.55, 2), (2.1, y))
        arrow(ax_bagging, (4.35, y), (4.9, y))
        arrow(ax_bagging, (6.9, y), (7.75, 2))
    box(ax_bagging, 8.8, 2, "$a(x) = \\frac{1}{B}\\,(b_1 + b_2 + \\dots + b_B)$", color=OCHRE + "33", edge=OCHRE, size=12)

    ax_stacking.set_xlim(0, 10)
    ax_stacking.set_ylim(0, 4)
    ax_stacking.axis("off")
    ax_stacking.set_title("стекинг: модели разных типов, ответы объединяет метамодель", fontsize=14, loc="left")
    box(ax_stacking, 0.9, 2, "обучающая\nвыборка", color=LIGHT)
    for y, name in [(3.3, "линейная регрессия $a_1$"), (2.0, "kNN $a_2$"), (0.7, "глубокое дерево $a_3$")]:
        box(ax_stacking, 3.4, y, name, color=BLUE + "33", edge=BLUE)
        arrow(ax_stacking, (1.55, 2), (2.25, y))
        arrow(ax_stacking, (4.6, y), (5.35, 2))
    box(ax_stacking, 6.3, 2, "предсказания\nвне выборки\n$a_1(x), a_2(x), a_3(x)$", color=OCHRE + "33", edge=OCHRE)
    arrow(ax_stacking, (7.25, 2), (7.85, 2))
    box(ax_stacking, 8.9, 2, "метамодель\n$a(x) = g(a_1(x), a_2(x), a_3(x))$", color=LIGHT, size=12)

    ax_boosting.set_xlim(0, 10)
    ax_boosting.set_ylim(0, 5)
    ax_boosting.axis("off")
    ax_boosting.set_title("бустинг: деревья обучают по очереди, каждое — на ошибках суммы предыдущих, ответ — сумма",
                          fontsize=14, loc="left")
    box(ax_boosting, 0.8, 1.0, "$a_0 = \\bar{y}$", color=LIGHT)
    step_x = [2.6, 5.0, 7.4]
    for t, x in enumerate(step_x, start=1):
        box(ax_boosting, x, 4.1, f"остатки\n$r_{t} = y - a_{t - 1}$", color=OCHRE + "33", edge=OCHRE)
        box(ax_boosting, x, 2.55, f"пень $b_{t}$,\nобученный на $r_{t}$", color=BLUE + "33", edge=BLUE)
        box(ax_boosting, x, 1.0, f"$a_{t} = a_{t - 1} + b_{t}$", color=LIGHT)
        arrow(ax_boosting, (x, 3.55), (x, 3.05))
        arrow(ax_boosting, (x, 2.05), (x, 1.4))
        previous_x = 0.8 if t == 1 else step_x[t - 2]
        arrow(ax_boosting, (previous_x + 0.45, 1.2), (x - 0.55, 3.75))
        arrow(ax_boosting, (previous_x + 0.55, 1.0), (x - 0.75, 1.0))
    box(ax_boosting, 9.4, 1.0, "…  $a_T = a_0 + b_1 + \\dots + b_T$", color=OCHRE + "33", edge=OCHRE, size=12)
    arrow(ax_boosting, (7.95, 1.0), (8.35, 1.0))

    plt.tight_layout()
    plt.show()


# --- бустинг на остатках --------------------------------------------------------
def stump_rule(stump, value_name=""):
    """Текст правила пня: вопрос и значения двух листьев."""
    structure = stump.tree_
    threshold = structure.threshold[0]
    left_value = structure.value[structure.children_left[0], 0, 0]
    right_value = structure.value[structure.children_right[0], 0, 0]
    rule = f"x ≤ {threshold:.2f}?\nда: {value_name}{left_value:+.2f}\nнет: {value_name}{right_value:+.2f}"
    return rule


def plot_boosting_steps(stumps, residuals_by_step, models_on_train, models_on_grid, train_mse,
                        x_train, y_train, x_grid, shown_steps=(1, 2, 3, 4)):
    """По столбцу на шаг: модель и ее остатки, пень на остатках; шкала по вертикали общая у всех графиков."""
    fig, axes = plt.subplots(2, len(shown_steps), figsize=(20, 8.6), sharex=True, sharey=True)

    for column, step in enumerate(shown_steps):
        stump = stumps[step - 1]
        residuals = residuals_by_step[step - 1]
        threshold = stump.tree_.threshold[0]

        ax = axes[0, column]
        ax.vlines(x_train[:, 0], models_on_train[step - 1], y_train, color=OCHRE, lw=1, alpha=0.8)
        ax.scatter(x_train[:, 0], y_train, s=12, color=GREY, zorder=3)
        ax.plot(x_grid[:, 0], models_on_grid[step - 1], color=BLUE, lw=2, label=f"$a_{step - 1}$")
        ax.set_title(f"шаг {step}: модель $a_{step - 1}$ и ее остатки", fontsize=12)
        ax.legend(loc="lower left")

        ax = axes[1, column]
        ax.axhline(0, color=GREY, lw=0.8)
        ax.scatter(x_train[:, 0], residuals, s=12, color=OCHRE, zorder=3, label=f"остатки $r_{step}$")
        ax.plot(x_grid[:, 0], stump.predict(x_grid), color=BLUE, lw=2.5, label=f"пень $b_{step}$")
        ax.axvline(threshold, color=BLUE, ls=":", lw=1)
        ax.text(0.97, 0.96, stump_rule(stump), transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=BLUE))
        ax.set_title(f"пень $b_{step}$ на остатках\nMSE на обучении: {train_mse[step - 1]:.3f} → {train_mse[step]:.3f}",
                     fontsize=12)
        ax.set_xlabel("x")
        ax.legend(loc="lower left")

    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("остаток")
    plt.tight_layout()
    plt.show()


def plot_boosting_progress(stumps, models_on_grid, train_mse, test_mse, x_train, y_train, x_grid,
                           noise_variance, shown_sums=(1, 6, 100)):
    """Первые пни по отдельности, их сумма и ошибка от числа пней."""
    fig, (ax_parts, ax_sum, ax_error) = plt.subplots(1, 3, figsize=(20, 5))

    part_colors = plt.cm.viridis(np.linspace(0, 0.85, 6))
    for number, (stump, color) in enumerate(zip(stumps[:6], part_colors), start=1):
        ax_parts.plot(x_grid[:, 0], stump.predict(x_grid), color=color, lw=2, label=f"$b_{{{number}}}$")
    ax_parts.axhline(0, color=GREY, lw=0.8)
    ax_parts.set_title("первые 6 пней по отдельности", fontsize=12)
    ax_parts.set_xlabel("x")
    ax_parts.legend(ncol=2, fontsize=9)

    ax_sum.scatter(x_train[:, 0], y_train, s=12, color=GREY, label="обучающая выборка")
    ax_sum.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), color=OCHRE, lw=2, ls="--", label="sin x")
    for T, color in zip(shown_sums, [GREY, BLUE, BLACK]):
        ax_sum.plot(x_grid[:, 0], models_on_grid[T], color=color, lw=1.8, label=f"$a_{{{T}}}$")
    ax_sum.set_title("сумма пней", fontsize=12)
    ax_sum.set_xlabel("x")
    ax_sum.legend(loc="lower left", fontsize=9)

    best_step = int(np.argmin(test_mse))
    steps = np.arange(len(train_mse))
    ax_error.plot(steps, train_mse, color=BLUE, lw=2, label="обучение")
    ax_error.plot(steps, test_mse, color=OCHRE, lw=2, label="тест")
    ax_error.axhline(noise_variance, color=GREY, ls="--", label=r"дисперсия шума $\sigma_\varepsilon^2$")
    ax_error.scatter(best_step, test_mse[best_step], color=BLACK, zorder=3, label=f"минимум на тесте: T = {best_step}")
    ax_error.set_yscale("log")
    ax_error.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
    ax_error.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    ax_error.minorticks_off()
    ax_error.set_xlabel("число пней T")
    ax_error.set_ylabel("MSE")
    ax_error.set_title("ошибка от числа пней", fontsize=12)
    ax_error.legend()

    plt.tight_layout()
    plt.show()


# --- градиентный бустинг --------------------------------------------------------
def plot_staged_curves(curves, xlabel, ylabel, ylim=None, xlim=None, noise_level=None, legend_outside=True):
    """Ошибка после 1, 2, ..., T деревьев; curves — {подпись: массив}; точка — минимум кривой."""
    colors = [OCHRE, GREY, BLUE, BLACK, "#4DAF4A"]
    plt.figure(figsize=(10, 4.8))
    for (label, curve), color in zip(curves.items(), colors):
        trees_axis = np.arange(1, len(curve) + 1)
        best_index = int(np.argmin(curve))
        plt.semilogx(trees_axis, curve, color=color, lw=1.8, label=label)
        plt.scatter(best_index + 1, curve[best_index], color=color, s=40, zorder=3)
    if noise_level is not None:
        plt.axhline(noise_level, color=GREY, ls=":", label=f"дисперсия шума $\\sigma_\\varepsilon^2 = {noise_level:g}$")
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_outside:
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    else:
        plt.legend()
    plt.show()


def plot_bars(values, xlabel, ylabel, highlight_first=True, horizontal=False, reference=None, reference_label=None):
    """Столбцы: values — {подпись: число}."""
    labels = list(values)
    numbers = list(values.values())
    plt.figure(figsize=(9, 6) if horizontal else (9, 4))
    if horizontal:
        colors = [OCHRE if "бустинг" in label or label in ("HistGradientBoostingRegressor", "LightGBM", "CatBoost",
                                                           "GradientBoostingRegressor") else GREY for label in labels]
        plt.barh(labels, numbers, color=colors)
        plt.gca().invert_yaxis()
        if reference is not None:
            plt.axvline(reference, color=BLACK, ls=":", label=reference_label)
            plt.legend()
    else:
        colors = [BLACK] + [OCHRE] * (len(labels) - 1) if highlight_first else [OCHRE] * len(labels)
        plt.bar(labels, numbers, color=colors)
        if reference is not None:
            plt.axhline(reference, color=BLACK, ls=":", lw=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# --- классификация ----------------------------------------------------------------
def plot_bias_variance_curve(stages, boosting_table, reference_row, noise_variance):
    """Слагаемые ошибки бустинга от числа пней; пунктир того же цвета — бэггинг."""
    plt.figure(figsize=(9, 4.4))
    for column, color in [("смещение²", BLUE), ("разброс", OCHRE), ("сумма трех слагаемых", BLACK)]:
        plt.semilogx(stages, boosting_table[column], "o-", color=color, label=f"бустинг: {column}")
        plt.axhline(reference_row[column], color=color, ls=":", lw=1.2)
    plt.axhline(noise_variance, color=GREY, ls="--", label=r"дисперсия шума $\sigma_\varepsilon^2$")
    plt.plot([], [], color=GREY, ls=":", label="бэггинг 100 пней (те же цвета)")
    plt.xticks(stages, stages)
    plt.minorticks_off()
    plt.xlabel("число пней T, темп η = 0.1")
    plt.ylabel("слагаемые ошибки, среднее по x")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.show()


def plot_gallery(entries, X_train, y_train, X_test, y_test):
    """Модели на двух лунах: сверху устройство, снизу вероятность класса 1.

    entries — список словарей: name, model (обученная), members (None для одного дерева или список деревьев),
    how (подпись способа объединения), accuracy.
    """
    X_all = np.vstack([X_train, X_test])
    x_values = np.linspace(X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5, 200)
    y_values = np.linspace(X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5, 200)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    extent = (x_values[0], x_values[-1], y_values[0], y_values[-1])

    n_blocks = (len(entries) + 2) // 3
    fig = plt.figure(figsize=(19, 10.5 * n_blocks))
    grid = fig.add_gridspec(2 * n_blocks, 3, height_ratios=[1, 1.15] * n_blocks, hspace=0.35, wspace=0.12)

    for index, entry in enumerate(entries):
        block, column = divmod(index, 3)
        ax_structure = fig.add_subplot(grid[2 * block, column])
        ax_map = fig.add_subplot(grid[2 * block + 1, column])
        model = entry["model"]

        if entry["members"] is None:
            n_leaves = model.get_n_leaves()
            draw_tree(ax_structure, model, detailed=n_leaves <= 8)
            ax_structure.set_title(f"{entry['name']}\nглубина {model.get_depth()}, листьев {n_leaves}", fontsize=12)
        else:
            ax_structure.axis("off")
            for k, member in enumerate(entry["members"][:3]):
                inset = ax_structure.inset_axes([0.02 + 0.33 * k, 0.12, 0.28, 0.78])
                draw_tree(inset, member, detailed=False)
                inset.set_title(f"$b_{k + 1}$: листьев {member.get_n_leaves()}", fontsize=10)
            ax_structure.text(0.5, 0.0, f"…  и еще {len(entry['members']) - 3} деревьев", transform=ax_structure.transAxes,
                              ha="center", fontsize=11, color=GREY)
            ax_structure.set_title(f"{entry['name']}\n{entry['how']}", fontsize=12)

        probability = model.predict_proba(grid_points)[:, 1].reshape(grid_x.shape)
        ax_map.imshow(probability, extent=extent, origin="lower", aspect="auto", cmap=PROBABILITY_COLORS,
                      vmin=0, vmax=1, alpha=0.6)
        ax_map.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=POINT_COLORS, s=14, edgecolor="white", lw=0.4)
        ax_map.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=POINT_COLORS, s=14, edgecolor=BLACK, lw=0.7)
        ax_map.text(0.97, 0.04, f"{entry['accuracy']:.2f}", transform=ax_map.transAxes, ha="right", fontsize=15,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.grid(False)

    plt.show()
