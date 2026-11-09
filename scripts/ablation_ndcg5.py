"""
Ablación del ranking evaluado con NDCG@5: separa el aporte de la corrección
XGBoost del aporte de la utilidad multicriterio.

Órdenes comparados sobre las mismas 10 viviendas de cada escenario:
  - Modelo: utilidad con el tiempo corregido (orden registrado en la evaluación).
  - Utilidad + OSRM: la misma utilidad con el tiempo OSRM sin corregir.
  - Solo tiempo XGBoost: tiempo corregido ascendente.
  - Solo tiempo OSRM y solo precio: los baselines del artículo.

La utilidad con OSRM se obtiene del puntaje registrado cambiando solo el término
de tiempo: S_osrm = S - f_tt(t_xgb) + f_tt(t_osrm). Los demás términos no
dependen del tiempo. Como la hoja guarda los tiempos en minutos enteros, el
análisis de sensibilidad repite el cálculo con tiempos perturbados ±0,5 min.

Uso:
    python scripts/ablation_ndcg5.py
"""

import csv
import math
from collections import defaultdict

import numpy as np
import openpyxl
from scipy.stats import wilcoxon

XLSX_PATH = "evaluation/ranking/NDCG5_scenarios_anonymised.xlsx"
SCORES_PATH = "evaluation/ranking/model_top10_per_scenario.csv"
OUT_PATH = "evaluation/ranking/ndcg5_ablation.md"

COL_ESC, COL_VIV, COL_PRECIO, COL_T_XGB, COL_T_OSRM, COL_MODELO, COL_HUMANO = 0, 6, 8, 11, 12, 13, 14
SEED = 42
N_SENS = 2000
N_BOOT = 10_000


def f_tt(t):
    """Término de tiempo de la utilidad evaluada (Eq. 5 del artículo)."""
    return 45 * (1 - t / 45) if t <= 45 else -min(20, ((t - 45) / 15) * 10)


def rel(h):
    return 1.0 if h <= 2 else 0.5 if h <= 5 else 0.0


def ndcg5(items, key):
    dcg = lambda orden: sum(it["rel"] / math.log2(i + 1) for i, it in enumerate(orden[:5], start=1))
    return dcg(sorted(items, key=key)) / dcg(sorted(items, key=lambda it: it["humano"]))


def cargar():
    scores = {(int(r["Escenario"]), r["Vivienda ID"]): float(r["Score"])
              for r in csv.DictReader(open(SCORES_PATH, encoding="utf-8"))}
    ws = openpyxl.load_workbook(XLSX_PATH, data_only=True)["Escenarios_Evaluacion"]
    esc = defaultdict(list)
    for r in ws.iter_rows(min_row=4, values_only=True):
        if r[COL_ESC] is None:
            continue
        esc[r[COL_ESC]].append({
            "viv": r[COL_VIV], "precio": r[COL_PRECIO], "t_xgb": r[COL_T_XGB], "t_osrm": r[COL_T_OSRM],
            "modelo": r[COL_MODELO], "humano": r[COL_HUMANO], "rel": rel(r[COL_HUMANO]),
            "score": scores[(r[COL_ESC], r[COL_VIV])],
        })
    for k, items in esc.items():
        assert len(items) == 10
        # El orden registrado del modelo debe ser el del puntaje registrado.
        por_modelo = sorted(items, key=lambda it: it["modelo"])
        assert all(a["score"] >= b["score"] for a, b in zip(por_modelo, por_modelo[1:])), k
    return dict(sorted(esc.items()))


def utilidad_osrm(items, dx=None, do=None):
    for i, it in enumerate(items):
        tx = it["t_xgb"] + (dx[i] if dx is not None else 0)
        to = it["t_osrm"] + (do[i] if do is not None else 0)
        it["s_osrm"] = min(100.0, max(0.0, it["score"] - f_tt(tx) + f_tt(to)))


def ordenes(items):
    return {
        "MiCasita (utility + XGBoost time)": ndcg5(items, lambda it: it["modelo"]),
        "Utility + uncorrected OSRM time": ndcg5(items, lambda it: (-it["s_osrm"], it["modelo"])),
        "XGBoost time only": ndcg5(items, lambda it: (it["t_xgb"], it["modelo"])),
        "Uncorrected OSRM time only": ndcg5(items, lambda it: (it["t_osrm"], it["modelo"])),
        "Price only": ndcg5(items, lambda it: (it["precio"], it["modelo"])),
    }


def holm(pvals):
    orden = sorted(range(len(pvals)), key=lambda i: pvals[i])
    ajust, acc = [0.0] * len(pvals), 0.0
    for rango, i in enumerate(orden):
        acc = max(acc, min(1.0, (len(pvals) - rango) * pvals[i]))
        ajust[i] = acc
    return ajust


def main():
    esc = cargar()
    for items in esc.values():
        utilidad_osrm(items)
    tabla = {k: [ordenes(items)[k] for items in esc.values()] for k in ordenes(next(iter(esc.values())))}
    nombres = list(tabla)
    base = np.array(tabla[nombres[0]])
    rng = np.random.default_rng(SEED)

    comparaciones = []
    for nombre in nombres[1:]:
        otro = np.array(tabla[nombre])
        d = base - otro
        nz = d[np.abs(d) > 1e-9]
        w = wilcoxon(nz, zero_method="wilcox", alternative="two-sided") if len(nz) else None
        boot = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
        comparaciones.append(dict(nombre=nombre, media=d.mean(), ci=np.percentile(boot, [2.5, 97.5]),
                                  mejor=int((d > 1e-9).sum()), igual=int((np.abs(d) <= 1e-9).sum()),
                                  peor=int((d < -1e-9).sum()), W=w.statistic if w else float("nan"),
                                  p=w.pvalue if w else 1.0, n=len(nz)))
    for c, p in zip(comparaciones, holm([c["p"] for c in comparaciones])):
        c["p_holm"] = p

    # Sensibilidad al redondeo de los tiempos a minutos enteros.
    sens_medias, sens_p, sens_mejor = [], [], []
    for _ in range(N_SENS):
        vals = []
        for items in esc.values():
            utilidad_osrm(items, rng.uniform(-0.5, 0.5, 10), rng.uniform(-0.5, 0.5, 10))
            vals.append(ndcg5(items, lambda it: (-it["s_osrm"], it["modelo"])))
        vals = np.array(vals)
        d = base - vals
        nz = d[np.abs(d) > 1e-9]
        sens_medias.append(vals.mean())
        sens_p.append(wilcoxon(nz).pvalue if len(nz) else 1.0)
        sens_mejor.append(int((d > 1e-9).sum()))
    for items in esc.values():
        utilidad_osrm(items)

    lines = ["# NDCG@5 ablation\n",
             f"Generated by `scripts/ablation_ndcg5.py` from `{XLSX_PATH}` and `{SCORES_PATH}` "
             f"({len(esc)} scenarios x 10 candidates).\n",
             "| Ordering | Mean NDCG@5 (SD) | Median | Min–max |", "|---|---|---|---|"]
    for nombre in nombres:
        v = np.array(tabla[nombre])
        lines.append(f"| {nombre} | {v.mean():.3f} ({v.std(ddof=1):.3f}) | {np.median(v):.3f} | {v.min():.3f}–{v.max():.3f} |")
    lines += ["", "Paired comparisons against MiCasita (two-sided Wilcoxon signed-rank, zero differences "
              "discarded, Holm correction over the four comparisons):", "",
              "| Comparison | Mean difference (95% CI) | Better / equal / worse | Non-zero pairs | W | p | Holm p |",
              "|---|---|---|---|---|---|---|"]
    for c in comparaciones:
        lines.append(f"| MiCasita vs. {c['nombre']} | {c['media']:.3f} ({c['ci'][0]:.3f}–{c['ci'][1]:.3f}) | "
                     f"{c['mejor']} / {c['igual']} / {c['peor']} | {c['n']} | {c['W']:.1f} | {c['p']:.4f} | {c['p_holm']:.4f} |")
    sm, sp = np.array(sens_medias), np.array(sens_p)
    lines += ["", f"Rounding sensitivity ({N_SENS} draws, times perturbed uniformly by ±0.5 min): mean NDCG@5 of "
              f"utility + OSRM {np.percentile(sm, 2.5):.3f}–{np.percentile(sm, 97.5):.3f} (2.5–97.5th percentile); "
              f"MiCasita better in {int(np.percentile(sens_mejor, 2.5))}–{int(np.percentile(sens_mejor, 97.5))} "
              f"scenarios; unadjusted p < 0.05 in {(sp < 0.05).mean():.1%} of draws.", "",
              "| Scenario | " + " | ".join(nombres) + " |", "|---|" + "---|" * len(nombres)]
    for i, k in enumerate(esc):
        lines.append(f"| {k} | " + " | ".join(f"{tabla[n][i]:.3f}" for n in nombres) + " |")
    open(OUT_PATH, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"Guardado: {OUT_PATH}")


if __name__ == "__main__":
    main()
