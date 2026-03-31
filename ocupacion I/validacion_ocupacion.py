import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# RUTAS
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
RESULTADOS_DIR = PROJECT_ROOT / "resultados_estocasticidad_ocupacion"
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(CURRENT_DIR))

from generar_ocupacion_2025_validacion import generar_ocupacion_2025_validacion


# ============================================================
# CONFIGURACIÓN
# ============================================================
N_OCUPANTES = 5
PERFIL = "P2"
SEEDS = [2025, 2026, 2027, 2028, 2029]
DIAS_VENTANA = 7


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def calcular_matriz_transicion(estados, n_estados):
    """
    Calcula la matriz de transición fila-normalizada:
    fila i = estado actual
    columna j = estado siguiente
    """
    conteos = np.zeros((n_estados, n_estados), dtype=float)

    for i in range(len(estados) - 1):
        actual = int(estados[i])
        siguiente = int(estados[i + 1])
        conteos[actual, siguiente] += 1

    matriz = conteos.copy()
    for i in range(n_estados):
        total_fila = matriz[i].sum()
        if total_fila > 0:
            matriz[i, :] /= total_fila

    return conteos, matriz


def anotar_heatmap(ax, matriz, fmt=".2f", umbral_texto=0.50):
    """
    Añade texto en cada celda del heatmap.
    """
    nrows, ncols = matriz.shape
    for i in range(nrows):
        for j in range(ncols):
            valor = matriz[i, j]
            color = "white" if valor >= umbral_texto else "black"
            ax.text(j, i, format(valor, fmt), ha="center", va="center", fontsize=9, color=color)


def guardar_csv(df, nombre):
    df.to_csv(RESULTADOS_DIR / nombre, index=False)


# ============================================================
# 1) GENERAR MÚLTIPLES REALIZACIONES
# ============================================================
simulaciones = []
matrices_por_seed = []
conteos_por_seed = []

for seed in SEEDS:
    df_occ, df_bin = generar_ocupacion_2025_validacion(
        n_ocupantes=N_OCUPANTES,
        perfil=PERFIL,
        seed=seed
    )

    df = df_occ.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["seed"] = seed
    df["hora"] = df["timestamp"].dt.hour
    df["fecha"] = df["timestamp"].dt.date
    simulaciones.append(df)

    estados = df["occ_total"].values
    conteos, matriz = calcular_matriz_transicion(estados, N_OCUPANTES + 1)
    conteos_por_seed.append(conteos)
    matrices_por_seed.append(matriz)

df_all = pd.concat(simulaciones, ignore_index=True)

print("Simulaciones generadas:", len(SEEDS))
print("Total de registros:", len(df_all))


# ============================================================
# 2) FIGURA 1: TRAYECTORIAS EN VENTANA CORTA
# ============================================================
fecha_inicio = df_all["timestamp"].min()
fecha_fin = fecha_inicio + pd.Timedelta(days=DIAS_VENTANA)

plt.figure(figsize=(15, 5))
for seed in SEEDS:
    df_s = df_all[
        (df_all["seed"] == seed) &
        (df_all["timestamp"] >= fecha_inicio) &
        (df_all["timestamp"] < fecha_fin)
    ].copy()

    plt.plot(
        df_s["timestamp"],
        df_s["occ_total"],
        label=f"Seed {seed}",
        linewidth=1.6
    )

plt.xlabel("Tiempo")
plt.ylabel("Ocupación total")
plt.title(f"Trayectorias de ocupación total en una ventana de {DIAS_VENTANA} días")
plt.legend(ncols=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "fig1_trayectorias_ocupacion.png", dpi=300)
plt.show()


# ============================================================
# 3) FIGURA 2: DISTRIBUCIÓN EMPÍRICA GLOBAL
# ============================================================
plt.figure(figsize=(8, 5))
bins = np.arange(-0.5, N_OCUPANTES + 1.5, 1)
plt.hist(
    df_all["occ_total"],
    bins=bins,
    edgecolor="black",
    density=True
)
plt.xticks(range(0, N_OCUPANTES + 1))
plt.xlabel("Estado de ocupación total")
plt.ylabel("Probabilidad empírica")
plt.title("Distribución empírica de la ocupación total")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "fig2_histograma_ocupacion.png", dpi=300)
plt.show()


# ============================================================
# 4) FIGURA 3: MEDIA ± STD POR HORA
# ============================================================
estad_hora = (
    df_all.groupby("hora")["occ_total"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)

x = estad_hora["hora"].values
media = estad_hora["mean"].values
std = estad_hora["std"].fillna(0).values

plt.figure(figsize=(10, 5))
plt.plot(x, media, marker="o", linewidth=2, label="Media horaria")
plt.fill_between(x, media - std, media + std, alpha=0.25, label="±1 desviación estándar")
plt.xticks(range(24))
plt.xlabel("Hora del día")
plt.ylabel("Ocupación total")
plt.title("Media y dispersión horaria de la ocupación total")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "fig3_media_std_por_hora.png", dpi=300)
plt.show()


# ============================================================
# 5) MATRICES DE TRANSICIÓN: PROMEDIO Y DESVIACIÓN ESTÁNDAR
# ============================================================
stack_matrices = np.stack(matrices_por_seed, axis=0)
matriz_media = np.mean(stack_matrices, axis=0)
matriz_std = np.std(stack_matrices, axis=0)

# Guardado numérico
pd.DataFrame(matriz_media).to_csv(RESULTADOS_DIR / "matriz_transicion_media.csv", index=False)
pd.DataFrame(matriz_std).to_csv(RESULTADOS_DIR / "matriz_transicion_std.csv", index=False)


# ============================================================
# 6) FIGURA 4: MATRIZ MEDIA DE TRANSICIÓN
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
img = ax.imshow(matriz_media, aspect="auto")
plt.colorbar(img, ax=ax, label="Probabilidad media")
ax.set_xticks(range(N_OCUPANTES + 1))
ax.set_yticks(range(N_OCUPANTES + 1))
ax.set_xlabel("Estado siguiente")
ax.set_ylabel("Estado actual")
ax.set_title("Matriz media de transición entre estados de ocupación")
anotar_heatmap(ax, matriz_media, fmt=".2f", umbral_texto=0.50)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "fig4_matriz_transicion_media.png", dpi=300)
plt.show()


# ============================================================
# 7) FIGURA 5: MATRIZ STD DE TRANSICIÓN
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
img = ax.imshow(matriz_std, aspect="auto")
plt.colorbar(img, ax=ax, label="Desviación estándar")
ax.set_xticks(range(N_OCUPANTES + 1))
ax.set_yticks(range(N_OCUPANTES + 1))
ax.set_xlabel("Estado siguiente")
ax.set_ylabel("Estado actual")
ax.set_title("Desviación estándar de la matriz de transición entre semillas")
anotar_heatmap(ax, matriz_std, fmt=".3f", umbral_texto=0.10)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "fig5_matriz_transicion_std.png", dpi=300)
plt.show()


# ============================================================
# 8) FIGURA 6: MATRICES INDIVIDUALES POR SEMILLA
# ============================================================
ncols = 2
nrows = int(np.ceil(len(SEEDS) / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 4 * nrows))
axes = np.array(axes).reshape(-1)

for ax, seed, matriz in zip(axes, SEEDS, matrices_por_seed):
    img = ax.imshow(matriz, aspect="auto")
    ax.set_title(f"Seed {seed}")
    ax.set_xticks(range(N_OCUPANTES + 1))
    ax.set_yticks(range(N_OCUPANTES + 1))
    ax.set_xlabel("Estado sig.")
    ax.set_ylabel("Estado act.")
    anotar_heatmap(ax, matriz, fmt=".2f", umbral_texto=0.50)

for k in range(len(SEEDS), len(axes)):
    axes[k].axis("off")

fig.suptitle("Matrices de transición por semilla", y=0.98)
fig.tight_layout()
plt.savefig(RESULTADOS_DIR / "fig6_matrices_por_semilla.png", dpi=300)
plt.show()


# ============================================================
# 9) MÉTRICAS DE ESTOCASTICIDAD
# ============================================================
p = df_all["occ_total"].value_counts(normalize=True).sort_index()
p = p.reindex(range(N_OCUPANTES + 1), fill_value=0)

eps = 1e-12
entropia = -np.sum(p.values * np.log(p.values + eps))
std_promedio = estad_hora["std"].mean()
p_vacia = p.loc[0] if 0 in p.index else 0.0

# Persistencia diagonal media
persistencia_media = np.mean(np.diag(matriz_media))

# Estado más frecuente
estado_modal = int(p.idxmax())
prob_estado_modal = float(p.max())

resumen = pd.DataFrame({
    "metrica": [
        "entropia_shannon",
        "std_promedio_por_hora",
        "probabilidad_vivienda_vacia",
        "ocupacion_media_global",
        "ocupacion_maxima_observada",
        "persistencia_media_diagonal",
        "estado_modal",
        "probabilidad_estado_modal"
    ],
    "valor": [
        entropia,
        std_promedio,
        p_vacia,
        df_all["occ_total"].mean(),
        df_all["occ_total"].max(),
        persistencia_media,
        estado_modal,
        prob_estado_modal
    ]
})

print("\n=== RESUMEN DE MÉTRICAS ===")
print(resumen)

guardar_csv(resumen, "resumen_estocasticidad.csv")
guardar_csv(estad_hora, "estadistica_horaria_ocupacion.csv")

# Distribución por hora y por semilla
dist_hora_seed = (
    df_all.groupby(["seed", "hora", "occ_total"])
    .size()
    .reset_index(name="conteo")
)
guardar_csv(dist_hora_seed, "distribucion_hora_seed.csv")

print(f"\nResultados guardados en: {RESULTADOS_DIR.resolve()}")