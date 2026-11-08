"""
Лабораторная работа № 8. Вариант 14.
Выбор заявок: жадные критерии «раннее окончание» и «максимальная ценность заявки»
против точного решения (динамическое программирование). 35 заявок.
Дополнительное задание Ж7: распределение ресурса между инициативами, цена ошибки.

Заявка = (начало, окончание, ценность). Заявки совместимы, если не пересекаются:
начало следующей >= окончания предыдущей. Цель: максимизировать суммарную ценность.
"""
import bisect
import json
import random
import statistics
import time

N = 35                  # число заявок по варианту
HORIZON = 100           # горизонт планирования, ч
TRIALS = 500            # число случайных наборов
SEED = 2024


# ---------------------------------------------------------------- данные
def make_requests(n, rng, horizon=HORIZON):
    """n заявок: начало — случайный момент, длительность 3…20 ч, ценность 10…100."""
    reqs = []
    for _ in range(n):
        start = rng.randint(0, horizon - 3)
        end = min(horizon + 20, start + rng.randint(3, 20))
        reqs.append((start, end, rng.randint(10, 100)))
    return reqs


# ------------------------------------------------------ жадные алгоритмы
def greedy_select(reqs, key):
    """Жадный алгоритм: заявки упорядочиваются по критерию key (по возрастанию),
    заявка принимается, если не пересекается с уже принятыми.
    Принятые заявки хранятся упорядоченно по началу, проверяются только соседи."""
    order = sorted(range(len(reqs)), key=key)
    starts, chosen = [], []              # упорядоченные начала принятых заявок
    by_start = {}
    for i in order:
        s, e, v = reqs[i]
        pos = bisect.bisect_left(starts, s)
        ok_left = pos == 0 or reqs[by_start[starts[pos - 1]]][1] <= s
        ok_right = pos == len(starts) or reqs[by_start[starts[pos]]][0] >= e
        if ok_left and ok_right:
            starts.insert(pos, s)
            by_start[s] = i
            chosen.append(i)
    return sum(reqs[i][2] for i in chosen), sorted(chosen)


def greedy_earliest_end(reqs):
    """Критерий 1: раннее окончание (при равенстве — раннее начало)."""
    return greedy_select(reqs, key=lambda i: (reqs[i][1], reqs[i][0]))


def greedy_max_value(reqs):
    """Критерий 2: максимальная ценность заявки (при равенстве — раннее окончание)."""
    return greedy_select(reqs, key=lambda i: (-reqs[i][2], reqs[i][1]))


# ----------------------------------------------------- точные алгоритмы
def exact_dp(reqs, weighted=True):
    """Точное решение: взвешенный выбор заявок. Заявки упорядочены по окончанию,
    p[j] — число заявок, закончившихся не позже начала j-й.
    D[j] = max(D[j−1], v_j + D[p[j]]). Трудоёмкость Θ(n log n)."""
    order = sorted(range(len(reqs)), key=lambda i: (reqs[i][1], reqs[i][0]))
    ends = [reqs[i][1] for i in order]
    w = [reqs[i][2] if weighted else 1 for i in order]
    p = [bisect.bisect_right(ends, reqs[i][0], 0, j) for j, i in enumerate(order)]
    D = [0] * (len(order) + 1)
    for j in range(1, len(order) + 1):
        D[j] = max(D[j - 1], w[j - 1] + D[p[j - 1]])
    chosen, j = [], len(order)
    while j > 0:                         # восстановление набора
        if D[j] == D[j - 1]:
            j -= 1
        else:
            chosen.append(order[j - 1])
            j = p[j - 1]
    return D[-1], sorted(chosen)


def exact_brute(reqs, weighted=True):
    """Полный перебор совместимых наборов (ветвление «взять / пропустить»).
    Возвращает (оптимум, число рассмотренных вершин перебора)."""
    order = sorted(range(len(reqs)), key=lambda i: reqs[i][0])
    best, nodes = 0, 0

    def rec(k, last_end, total):
        nonlocal best, nodes
        nodes += 1
        if k == len(order):
            best = max(best, total)
            return
        s, e, v = reqs[order[k]]
        rec(k + 1, last_end, total)                          # пропустить
        if s >= last_end:                                    # взять
            rec(k + 1, e, total + (v if weighted else 1))

    rec(0, -1, 0)
    return best, nodes


# --------------------------------------------------------------- серии
def experiment(trials=TRIALS, n=N, seed=SEED):
    """Отклонение жадных решений от оптимума на серии случайных наборов."""
    rng = random.Random(seed)
    names = ["Раннее окончание", "Максимальная ценность"]
    dev = {k: [] for k in names}
    dev_count = {k: [] for k in names}
    hits = {k: 0 for k in names}
    hits_count = {k: 0 for k in names}
    losses = []
    for _ in range(trials):
        reqs = make_requests(n, rng)
        opt, _ = exact_dp(reqs)
        opt_cnt, _ = exact_dp(reqs, weighted=False)
        for name, func in zip(names, (greedy_earliest_end, greedy_max_value)):
            got, chosen = func(reqs)
            dev[name].append((opt - got) / opt * 100)
            hits[name] += got == opt
            cnt = len(chosen)
            dev_count[name].append((opt_cnt - cnt) / opt_cnt * 100)
            hits_count[name] += cnt == opt_cnt
        losses.append(opt - greedy_earliest_end(reqs)[0])
    return dev, hits, dev_count, hits_count


def verify_dp_against_brute(sets=200, n=15, seed=5):
    """Сверка ДП с полным перебором на малых наборах (обе постановки)."""
    rng = random.Random(seed)
    for _ in range(sets):
        reqs = make_requests(n, rng)
        assert exact_dp(reqs)[0] == exact_brute(reqs)[0]
        assert exact_dp(reqs, weighted=False)[0] == exact_brute(reqs, weighted=False)[0]
        assert greedy_earliest_end(reqs)[0] <= exact_dp(reqs)[0]
    return sets


def counterexamples():
    """Контрпримеры, построенные вручную."""
    ex1 = [(0, 5, 10), (1, 6, 100), (6, 9, 20)]            # ломает «раннее окончание»
    ex2 = [(0, 10, 50), (0, 5, 30), (5, 10, 30)]           # ломает «максимальную ценность»
    out = []
    for reqs in (ex1, ex2):
        g1, c1 = greedy_earliest_end(reqs)
        g2, c2 = greedy_max_value(reqs)
        opt, co = exact_dp(reqs)
        out.append({"reqs": reqs, "end": (g1, c1), "value": (g2, c2), "opt": (opt, co),
                    "count_opt": exact_dp(reqs, weighted=False)[0],
                    "count_value": len(c2), "count_end": len(c1)})
    return out


# ----------------------------------------------------------------- время
def timing():
    rng = random.Random(1)
    rows = []
    for n in (35, 100, 1000, 10_000, 100_000):
        reqs = make_requests(n, rng, horizon=HORIZON * n // N)     # плотность постоянна
        row = {"n": n}
        for name, func in (("greedy_end", greedy_earliest_end), ("greedy_value", greedy_max_value),
                           ("dp", exact_dp)):
            reps = 200 if n <= 1000 else 3
            t0 = time.perf_counter()
            for _ in range(reps):
                func(reqs)
            row[name] = (time.perf_counter() - t0) / reps
        rows.append(row)
    brute = []
    rng = random.Random(2)
    for n in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80):
        reqs = make_requests(n, rng)
        t0 = time.perf_counter()
        _, nodes = exact_brute(reqs)
        t = time.perf_counter() - t0
        brute.append({"n": n, "t": t, "nodes": nodes})
        if t > 10:
            break
    return rows, brute


# ------------------------------------------------------------- Ж7: бюджет
def application_case(seed=14):
    """Одна производственная линия (единственный ресурс), 35 инициатив запрашивают
    интервалы её использования; ценность заявки — прибыль, тыс. руб."""
    rng = random.Random(seed)
    reqs = make_requests(N, rng)
    opt, c_opt = exact_dp(reqs)
    g1, c1 = greedy_earliest_end(reqs)
    g2, c2 = greedy_max_value(reqs)
    return {"reqs": reqs, "opt": (opt, c_opt), "end": (g1, c1), "value": (g2, c2)}


# ----------------------------------------------------------------- графики
def make_plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, d) in zip(axes, res["dev"].items()):
        ax.hist(d, bins=30, color="#4c72b0" if name.startswith("Ран") else "#c44e52")
        ax.axvline(statistics.mean(d), color="k", ls="--", label=f"среднее {statistics.mean(d):.1f} %")
        ax.axvline(statistics.median(d), color="g", ls=":", label=f"медиана {statistics.median(d):.1f} %")
        ax.set_title(name); ax.set_xlabel("отклонение от оптимума, %"); ax.set_ylabel("число наборов")
        ax.legend(); ax.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig("img/lr8_hist.png", dpi=170); plt.close()

    t = res["timing"]
    plt.figure(figsize=(6.6, 4.2))
    ns = [r["n"] for r in t]
    for key, label, mk in (("greedy_end", "жадный (раннее окончание)", "o-"),
                           ("greedy_value", "жадный (макс. ценность)", "s-"),
                           ("dp", "точный (ДП)", "^-")):
        plt.loglog(ns, [r[key] for r in t], mk, label=label)
    plt.xlabel("число заявок n"); plt.ylabel("время, с"); plt.grid(True, which="both", alpha=.3)
    plt.title("Время работы алгоритмов"); plt.legend()
    plt.tight_layout(); plt.savefig("img/lr8_time.png", dpi=170); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, ex, title in zip(axes, res["counter"], ("Контрпример 1 (критерий «раннее окончание»)",
                                                    "Контрпример 2 (критерий «максимальная ценность»)")):
        for k, (s, e, v) in enumerate(ex["reqs"]):
            col = "#2ca02c" if k in ex["opt"][1] else "#bbbbbb"
            ax.barh(k, e - s, left=s, color=col, edgecolor="k")
            ax.text(s + (e - s) / 2, k, f"v={v}", ha="center", va="center", fontsize=9)
        ax.set_yticks(range(len(ex["reqs"])))
        ax.set_yticklabels([f"заявка {k + 1}" for k in range(len(ex["reqs"]))])
        ax.invert_yaxis(); ax.set_xlabel("время, ч"); ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", alpha=.3)
    plt.tight_layout(); plt.savefig("img/lr8_counter.png", dpi=170); plt.close()


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    res = {}
    print("Сверка ДП с перебором:", verify_dp_against_brute(), "наборов — совпало")

    res["counter"] = counterexamples()
    for ex in res["counter"]:
        print("Контрпример", ex["reqs"], "жадно(окончание)", ex["end"], "жадно(ценность)",
              ex["value"], "оптимум", ex["opt"])

    dev, hits, dev_c, hits_c = experiment()
    res["dev"], res["hits"], res["dev_count"], res["hits_count"] = dev, hits, dev_c, hits_c
    for name in dev:
        d = dev[name]
        print(f"{name:<24} среднее {statistics.mean(d):6.2f} медиана {statistics.median(d):6.2f} "
              f"макс {max(d):6.2f} точных {hits[name] / TRIALS * 100:5.1f} %  "
              f"| по числу заявок: среднее {statistics.mean(dev_c[name]):.2f} макс {max(dev_c[name]):.2f} "
              f"точных {hits_c[name] / TRIALS * 100:.1f} %")

    res["timing"], res["brute"] = timing()
    for r in res["timing"]:
        print(r)
    print(res["brute"])

    res["app"] = application_case()
    # цена ошибки по серии сценариев (ценность в тыс. руб.)
    rng = random.Random(SEED)
    loss_end, loss_val = [], []
    for _ in range(TRIALS):
        reqs = make_requests(N, rng)
        opt = exact_dp(reqs)[0]
        loss_end.append(opt - greedy_earliest_end(reqs)[0])
        loss_val.append(opt - greedy_max_value(reqs)[0])
    res["loss"] = {"end": loss_end, "value": loss_val}
    print("потери, тыс. руб.: окончание", statistics.mean(loss_end), max(loss_end),
          "ценность", statistics.mean(loss_val), max(loss_val))

    with open("res/lr8.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    make_plots(res)
