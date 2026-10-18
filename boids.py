"""
Лабораторная работа. Вариант №1: моделирование стаи птиц (алгоритм Boids, К. Рейнольдс).

Запуск:
    python boids.py                       # интерактивная анимация со слайдерами
    python boids.py --boundary reflect    # отражение от границ вместо циклического перехода
    python boids.py --experiments         # серия экспериментов: таблица + скриншоты для отчёта

Зависимости: numpy, matplotlib   (pip install numpy matplotlib)
"""
import argparse
import csv
import os
from dataclasses import dataclass, replace

import numpy as np
import matplotlib

# ---------------------------------------------------------------------------
# 1. ПАРАМЕТРЫ (всё, что можно менять, лежит здесь)
# ---------------------------------------------------------------------------


@dataclass
class EnvParams:
    """Параметры среды."""
    width: float = 100.0
    height: float = 100.0
    boundary: str = "wrap"          # "wrap" - циклический переход, "reflect" - отражение


@dataclass
class AgentParams:
    """Параметры агента."""
    n_agents: int = 50              # количество птиц (по заданию >= 20)
    max_speed: float = 2.0          # максимальная скорость
    min_speed: float = 0.5          # минимальная скорость (птица не зависает на месте)
    max_accel: float = 0.1          # максимальное ускорение
    perception_radius: float = 15.0  # радиус взаимодействия (поиск соседей)
    separation_radius: float = 4.0  # радиус, в котором птицы "отталкиваются"


@dataclass
class Weights:
    """Веса правил Рейнольдса."""
    cohesion: float = 0.01          # центрирование
    alignment: float = 0.05         # выравнивание
    separation: float = 0.1         # разделение


# ---------------------------------------------------------------------------
# 2. МОДЕЛЬ
# ---------------------------------------------------------------------------


def limit(vectors, max_len):
    """Ограничивает длину каждого вектора (строки массива) значением max_len."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_len / np.maximum(norm, 1e-9))
    return vectors * scale


class Flock:
    """Стая: хранит положения и скорости всех агентов и делает шаг моделирования."""

    def __init__(self, env: EnvParams, agent: AgentParams, weights: Weights, seed=None):
        self.env, self.agent, self.w = env, agent, weights
        self.rng = np.random.default_rng(seed)
        n = agent.n_agents
        self.size = np.array([env.width, env.height])
        # начальные положения: случайно по всей плоскости
        self.pos = self.rng.uniform([0, 0], self.size, size=(n, 2))
        # начальные скорости: случайное направление и скорость
        angle = self.rng.uniform(0, 2 * np.pi, n)
        speed = self.rng.uniform(agent.min_speed, agent.max_speed, n)
        self.vel = np.column_stack((np.cos(angle), np.sin(angle))) * speed[:, None]
        self.step_no = 0

    # --- вспомогательное ---------------------------------------------------
    def _pair_vectors(self):
        """diff[i, j] = вектор от агента i к агенту j (с учётом тора при wrap)."""
        diff = self.pos[None, :, :] - self.pos[:, None, :]
        if self.env.boundary == "wrap":
            diff -= self.size * np.round(diff / self.size)   # кратчайший путь через границу
        dist = np.linalg.norm(diff, axis=2)
        return diff, dist

    # --- правила Boids -----------------------------------------------------
    def compute_acceleration(self):
        a, w = self.agent, self.w
        diff, dist = self._pair_vectors()
        np.fill_diagonal(dist, np.inf)                     # агент не сосед самому себе
        neigh = dist < a.perception_radius                 # матрица соседей (N x N)
        count = neigh.sum(axis=1, keepdims=True)           # число соседей
        has = count > 0
        safe_count = np.maximum(count, 1)

        # 1) Центрирование: вектор к центру масс соседей
        cohesion = (diff * neigh[:, :, None]).sum(axis=1) / safe_count
        cohesion = np.where(has, cohesion, 0.0)

        # 2) Выравнивание: разница между средней скоростью соседей и своей
        mean_vel = (neigh.astype(float) @ self.vel) / safe_count
        alignment = np.where(has, mean_vel - self.vel, 0.0)

        # 3) Разделение: уход от слишком близких соседей
        close = dist < a.separation_radius
        separation = -(diff * close[:, :, None]).sum(axis=1)

        # Взвешенная сумма + ограничение ускорения
        acc = w.cohesion * cohesion + w.alignment * alignment + w.separation * separation
        return limit(acc, a.max_accel)

    # --- границы -----------------------------------------------------------
    def apply_boundary(self):
        if self.env.boundary == "wrap":
            self.pos %= self.size
        else:  # reflect
            for d in (0, 1):
                low = self.pos[:, d] < 0
                high = self.pos[:, d] > self.size[d]
                self.pos[low, d] *= -1
                self.pos[high, d] = 2 * self.size[d] - self.pos[high, d]
                self.vel[low | high, d] *= -1

    # --- шаг моделирования -------------------------------------------------
    def step(self):
        self.vel += self.compute_acceleration()
        # ограничение скорости сверху и снизу
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        speed_clipped = np.clip(speed, self.agent.min_speed, self.agent.max_speed)
        self.vel *= speed_clipped / np.maximum(speed, 1e-9)
        self.pos += self.vel
        self.apply_boundary()
        self.step_no += 1

    # --- метрики для экспериментов ----------------------------------------
    def metrics(self, link_radius=None, collision_dist=1.0):
        diff, dist = self._pair_vectors()
        np.fill_diagonal(dist, np.inf)
        n = len(self.pos)
        unit = self.vel / np.maximum(np.linalg.norm(self.vel, axis=1, keepdims=True), 1e-9)
        polarization = np.linalg.norm(unit.mean(axis=0))          # 1 = все летят в одну сторону
        nn_dist = dist.min(axis=1).mean()                         # среднее расстояние до ближайшего
        collisions = int((dist < collision_dist).sum() // 2)      # пар ближе collision_dist
        # число групп: компоненты связности графа "ближе link_radius"
        link = link_radius or self.agent.perception_radius
        adj = dist < link
        seen, groups = np.zeros(n, bool), 0
        for s in range(n):
            if seen[s]:
                continue
            groups += 1
            stack = [s]
            seen[s] = True
            while stack:
                i = stack.pop()
                for j in np.where(adj[i] & ~seen)[0]:
                    seen[j] = True
                    stack.append(j)
        return dict(polarization=polarization, nn_dist=nn_dist, collisions=collisions, groups=groups)


# ---------------------------------------------------------------------------
# 3. ВИЗУАЛИЗАЦИЯ (интерактивная анимация со слайдерами)
# ---------------------------------------------------------------------------


def draw_snapshot(flock, ax, title=""):
    ax.clear()
    ax.set_xlim(0, flock.env.width)
    ax.set_ylim(0, flock.env.height)
    ax.set_aspect("equal")
    ax.quiver(flock.pos[:, 0], flock.pos[:, 1], flock.vel[:, 0], flock.vel[:, 1],
              color="tab:blue", pivot="mid", scale=40, width=0.005)
    ax.set_title(title)


def run_animation(env, agent, weights, seed=None, interval=30):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Slider

    flock = Flock(env, agent, weights, seed)
    fig, ax = plt.subplots(figsize=(8, 9))
    plt.subplots_adjust(bottom=0.30)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    q = ax.quiver(flock.pos[:, 0], flock.pos[:, 1], flock.vel[:, 0], flock.vel[:, 1],
                  color="tab:blue", pivot="mid", scale=40, width=0.005)
    info = ax.text(0.01, 1.01, "", transform=ax.transAxes)

    # слайдеры: меняем параметры прямо во время работы
    def add_slider(y, label, vmin, vmax, init, setter):
        s = Slider(plt.axes([0.15, y, 0.7, 0.03]), label, vmin, vmax, valinit=init)
        s.on_changed(setter)
        return s

    sliders = [
        add_slider(0.20, "Радиус", 1, 50, agent.perception_radius,
                   lambda v: setattr(agent, "perception_radius", v)),
        add_slider(0.15, "Центриров.", 0, 0.1, weights.cohesion,
                   lambda v: setattr(weights, "cohesion", v)),
        add_slider(0.10, "Выравнив.", 0, 0.2, weights.alignment,
                   lambda v: setattr(weights, "alignment", v)),
        add_slider(0.05, "Разделение", 0, 0.5, weights.separation,
                   lambda v: setattr(weights, "separation", v)),
    ]

    def update(_):
        flock.step()
        q.set_offsets(flock.pos)
        q.set_UVC(flock.vel[:, 0], flock.vel[:, 1])
        m = flock.metrics()
        info.set_text(f"шаг {flock.step_no} | поляризация {m['polarization']:.2f} | групп {m['groups']}")
        return q, info

    anim = FuncAnimation(fig, update, interval=interval, blit=False, cache_frame_data=False)
    fig._keep = (anim, sliders)     # чтобы объекты не удалил сборщик мусора
    plt.show()


# ---------------------------------------------------------------------------
# 4. ЭКСПЕРИМЕНТЫ
# ---------------------------------------------------------------------------


def run_experiments(env, agent, weights, steps=500, seed=1, out_dir="results"):
    """Серия запусков с разными параметрами. Результат: results.csv + скриншоты."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    W = Weights
    experiments = [
        # (название, параметры агента, веса)
        ("Базовый (R=15, 0.01/0.05/0.1)", agent, weights),
        ("Малый радиус R=5", replace(agent, perception_radius=5), weights),
        ("Средний радиус R=10", replace(agent, perception_radius=10), weights),
        ("Большой радиус R=30", replace(agent, perception_radius=30), weights),
        ("Огромный радиус R=60", replace(agent, perception_radius=60), weights),
        ("Сильное центрирование 0.05", agent, W(0.05, 0.05, 0.1)),
        ("Без центрирования 0", agent, W(0.0, 0.05, 0.1)),
        ("Сильное выравнивание 0.2", agent, W(0.01, 0.2, 0.1)),
        ("Без выравнивания 0", agent, W(0.01, 0.0, 0.1)),
        ("Слабое разделение 0.01", agent, W(0.01, 0.05, 0.01)),
        ("Без разделения 0", agent, W(0.01, 0.05, 0.0)),
        ("Сильное разделение 0.5", agent, W(0.01, 0.05, 0.5)),
    ]

    rows = []
    for idx, (name, ag, wt) in enumerate(experiments, 1):
        flock = Flock(env, ag, replace(wt), seed)
        history = []
        for _ in range(steps):
            flock.step()
            history.append(flock.metrics())
        tail = history[-100:]                       # усредняем по последним 100 шагам
        row = {
            "№": idx, "эксперимент": name, "радиус": ag.perception_radius,
            "центриров.": wt.cohesion, "выравнив.": wt.alignment, "разделение": wt.separation,
            "поляризация": round(float(np.mean([m["polarization"] for m in tail])), 3),
            "ср.расст. до ближайшего": round(float(np.mean([m["nn_dist"] for m in tail])), 2),
            "столкновений (пар<1)": round(float(np.mean([m["collisions"] for m in tail])), 1),
            "групп": round(float(np.mean([m["groups"] for m in tail])), 1),
        }
        rows.append(row)

        fig, ax = plt.subplots(figsize=(5, 5))
        draw_snapshot(flock, ax, f"{idx}. {name}\nшаг {steps}")
        fig.savefig(os.path.join(out_dir, f"exp_{idx:02d}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    # красивый вывод в консоль
    keys = list(rows[0].keys())
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))
    print(f"\nГотово: {out_dir}/results.csv и скриншоты {out_dir}/exp_XX.png")


# ---------------------------------------------------------------------------
# 5. ТОЧКА ВХОДА
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Boids: стая птиц")
    p.add_argument("--n", type=int, default=AgentParams.n_agents, help="число птиц")
    p.add_argument("--radius", type=float, default=AgentParams.perception_radius)
    p.add_argument("--boundary", choices=["wrap", "reflect"], default="wrap")
    p.add_argument("--cohesion", type=float, default=Weights.cohesion)
    p.add_argument("--alignment", type=float, default=Weights.alignment)
    p.add_argument("--separation", type=float, default=Weights.separation)
    p.add_argument("--steps", type=int, default=500, help="шагов в экспериментах")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--experiments", action="store_true", help="серия экспериментов без окна")
    args = p.parse_args()

    env = EnvParams(boundary=args.boundary)
    agent = AgentParams(n_agents=args.n, perception_radius=args.radius)
    weights = Weights(args.cohesion, args.alignment, args.separation)

    if args.experiments:
        run_experiments(env, agent, weights, steps=args.steps, seed=args.seed or 1)
    else:
        run_animation(env, agent, weights, seed=args.seed)


if __name__ == "__main__":
    main()
