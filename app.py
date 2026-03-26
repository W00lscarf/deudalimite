import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def help_text(text: str) -> str:
    return text

st.set_page_config(page_title="Simulador de deuda prudente", layout="wide")

# =========================
# Utilidades del modelo
# =========================

def simulate_paths(
    d0,
    horizon,
    n_sims,
    g_mean,
    pi_mean,
    i_mean,
    u_mean,
    d_bar,
    debt_ceiling,
    BPMF,
    alpha,
    beta,
    required_probability,
    g_std,
    i_std,
    pb_shock_std,
    u_std,
    seed,
):
    rng = np.random.default_rng(seed)

    debt = np.zeros((n_sims, horizon + 1))
    pb = np.zeros((n_sims, horizon))
    debt[:, 0] = d0

    for t in range(1, horizon + 1):
        g = rng.normal(g_mean, g_std, n_sims)
        pi = rng.normal(pi_mean, 0.01, n_sims)
        i = rng.normal(i_mean, i_std, n_sims)
        u = rng.normal(u_mean, u_std, n_sims)

        eps_pb = rng.normal(0, pb_shock_std, n_sims)
        pb_t = alpha + beta * (debt[:, t - 1] - d_bar) + eps_pb
        pb_t = np.clip(pb_t, -0.08, 0.04)
        pb[:, t - 1] = pb_t

        factor = (1 + i) / ((1 + g) * (1 + pi))
        debt[:, t] = factor * debt[:, t - 1] - pb_t + u

    cond_pb = (pb <= BPMF).all(axis=1)
    cond_debt = (debt[:, 1:] < debt_ceiling).all(axis=1)
    prob_ok = (cond_pb & cond_debt).mean()

    return prob_ok, debt, pb


def prudent_level(grid, **params):
    results = []
    for d0 in grid:
        prob_ok, _, _ = simulate_paths(d0=d0, **params)
        results.append((d0, prob_ok))

    df = pd.DataFrame(results, columns=["deuda_neta_inicial", "probabilidad_sostenible"])
    candidates = df[df["probabilidad_sostenible"] >= params["required_probability"]]
    prudent = candidates["deuda_neta_inicial"].max() if len(candidates) else np.nan
    return prudent, df


def run_reference_path(d0, years, **params):
    prob_ok, debt, pb = simulate_paths(d0=d0, **params)
    p50 = np.percentile(debt, 50, axis=0) * 100
    p10 = np.percentile(debt, 10, axis=0) * 100
    p90 = np.percentile(debt, 90, axis=0) * 100
    pb50 = np.percentile(pb, 50, axis=0) * 100
    return prob_ok, p10, p50, p90, pb50


def pct(x):
    return f"{x * 100:.1f}%"


# =========================
# Interfaz
# =========================

st.title("Simulador de deuda prudente estilo CFA (simplificado)")
st.markdown(
    """
Esta app permite **mover las palancas** del modelo para ver cómo cambia el **nivel prudente de deuda neta**.

La lógica es simple: se busca el mayor nivel inicial de deuda que, bajo muchas simulaciones, siga siendo compatible con una trayectoria fiscal sostenible.
No replica exactamente el modelo oficial del CFA, pero sí su intuición: **deuda + crecimiento + tasa de interés + reacción fiscal + shocks**.
"""
)

with st.sidebar:
    st.header("Parámetros del modelo")

    st.subheader("Tamaño del ejercicio")
    horizon = st.slider("Horizonte (años)", 5, 20, 10)
    n_sims = st.slider("Número de simulaciones", 1000, 20000, 6000, step=1000)
    seed = st.number_input("Semilla aleatoria", min_value=0, max_value=999999, value=42)

    st.subheader("Supuestos macro")
    g_mean = st.slider(
        "Crecimiento real promedio",
        0.0,
        0.05,
        0.02,
        step=0.001,
        help=help_text("Tasa promedio de expansión real de la economía. Si sube, la deuda pesa menos respecto del PIB y el límite prudente tiende a aumentar."),
    )
    pi_mean = st.slider(
        "Inflación / deflactor promedio",
        0.0,
        0.08,
        0.03,
        step=0.001,
        help=help_text("Aproxima el crecimiento nominal adicional del PIB vía precios. Más inflación reduce mecánicamente la razón deuda/PIB, aunque en la realidad puede venir acompañada de otras tensiones."),
    )
    i_mean = st.slider(
        "Tasa de interés nominal promedio",
        0.0,
        0.10,
        0.05,
        step=0.001,
        help=help_text("Costo promedio de financiamiento de la deuda. Si sube, el servicio de la deuda se encarece y el límite prudente suele bajar."),
    )
    u_mean = st.slider(
        "Otros ajustes promedio",
        -0.01,
        0.02,
        0.003,
        step=0.001,
        help=help_text("Recoge necesidades de financiamiento que no pasan directamente por el balance primario: movimientos de caja, capitalizaciones, uso o acumulación de activos del Tesoro, entre otros."),
    )

    st.subheader("Volatilidad")
    g_std = st.slider(
        "Volatilidad del crecimiento",
        0.0,
        0.05,
        0.012,
        step=0.001,
        help=help_text("Mide qué tan inestable es el crecimiento real. Más volatilidad implica más probabilidad de años malos y, por tanto, un menor límite prudente."),
    )
    i_std = st.slider(
        "Volatilidad de la tasa de interés",
        0.0,
        0.05,
        0.010,
        step=0.001,
        help=help_text("Refleja cuán incierto es el costo financiero de la deuda. Si aumenta, también sube el riesgo de trayectorias fiscales adversas."),
    )
    pb_shock_std = st.slider(
        "Volatilidad fiscal",
        0.0,
        0.03,
        0.006,
        step=0.001,
        help=help_text("Captura shocks sobre el balance primario: caídas de recaudación, aumentos inesperados de gasto o deterioros cíclicos."),
    )
    u_std = st.slider(
        "Volatilidad de otros ajustes",
        0.0,
        0.03,
        0.003,
        step=0.001,
        help=help_text("Representa la incertidumbre asociada a ajustes fuera del balance primario, como capitalizaciones, caja o movimientos patrimoniales."),
    )

    st.subheader("Comportamiento fiscal")
    BPMF = st.slider(
        "Balance primario máximo factible (BPMF)",
        0.0,
        0.05,
        0.0225,
        step=0.0025,
        help=help_text("Es el mayor superávit primario que el Estado podría sostener de manera realista. Si sube, el país tendría más capacidad de ajuste frente a una deuda alta."),
    )
    alpha = st.slider(
        "Intercepto fiscal (α)",
        -0.01,
        0.02,
        0.003,
        step=0.001,
        help=help_text("Es la posición fiscal de base del modelo. Un α más alto significa que, incluso sin presión de la deuda, el balance primario tiende a ser mejor."),
    )
    beta = st.slider(
        "Reacción fiscal a la deuda (β)",
        0.0,
        0.20,
        0.08,
        step=0.005,
        help=help_text("Mide cuánto corrige el gobierno cuando la deuda sube. Si β aumenta, la política fiscal reacciona con más fuerza y el límite prudente suele elevarse."),
    )
    d_bar = st.slider(
        "Nivel de referencia de deuda",
        0.10,
        0.60,
        0.30,
        step=0.01,
        help=help_text("Es el nivel a partir del cual el modelo considera que la deuda ya es relevante para activar una reacción fiscal más intensa."),
    )

    st.subheader("Criterio de prudencia")
    required_probability = st.slider(
        "Probabilidad mínima exigida",
        0.50,
        0.99,
        0.83,
        step=0.01,
        help=help_text("Define qué tan estricto eres para declarar prudente un nivel de deuda. Si la elevas, exiges más seguridad y el límite prudente tiende a caer."),
    )
    debt_ceiling = st.slider(
        "Techo de deuda en la simulación",
        0.40,
        1.50,
        0.80,
        step=0.05,
        help=help_text("Es una barrera de seguridad del ejercicio. Si una trayectoria supera este nivel, se considera que dejó de ser razonable o estable."),
    )

    st.subheader("Búsqueda del límite")
    grid_min = st.slider(
        "Mínimo deuda inicial a probar",
        0.10,
        0.50,
        0.20,
        step=0.01,
        help=help_text("Límite inferior del rango donde la app buscará el nivel prudente."),
    )
    grid_max = st.slider(
        "Máximo deuda inicial a probar",
        0.20,
        0.90,
        0.50,
        step=0.01,
        help=help_text("Límite superior del rango donde la app buscará el nivel prudente. Si el resultado queda demasiado bajo o alto, conviene ampliar este rango."),
    )
    grid_step = st.slider(
        "Paso de búsqueda",
        0.001,
        0.02,
        0.005,
        step=0.001,
        help=help_text("Precisión de la búsqueda. Pasos más pequeños entregan un resultado más fino, pero hacen más lenta la app."),
    )

params = {
    "horizon": horizon,
    "n_sims": n_sims,
    "g_mean": g_mean,
    "pi_mean": pi_mean,
    "i_mean": i_mean,
    "u_mean": u_mean,
    "d_bar": d_bar,
    "debt_ceiling": debt_ceiling,
    "BPMF": BPMF,
    "alpha": alpha,
    "beta": beta,
    "required_probability": required_probability,
    "g_std": g_std,
    "i_std": i_std,
    "pb_shock_std": pb_shock_std,
    "u_std": u_std,
    "seed": seed,
}

grid = np.arange(grid_min, grid_max + grid_step, grid_step)
years = np.arange(0, horizon + 1)

prudent, df_prob = prudent_level(grid=grid, **params)

col1, col2, col3 = st.columns(3)

if np.isnan(prudent):
    col1.metric("Nivel prudente estimado", "No encontrado")
    col2.metric("Probabilidad exigida", f"{required_probability * 100:.0f}%")
    col3.metric("BPMF", f"{BPMF * 100:.2f}% del PIB")
    st.warning("Con estos supuestos, ningún nivel del rango evaluado cumple el criterio de prudencia. Amplía el rango o relaja los supuestos.")
else:
    col1.metric("Nivel prudente estimado", f"{prudent * 100:.1f}% del PIB")
    col2.metric("Probabilidad exigida", f"{required_probability * 100:.0f}%")
    col3.metric("BPMF", f"{BPMF * 100:.2f}% del PIB")

st.divider()

left, right = st.columns([1.1, 1])

with left:
    st.subheader("¿Qué significa cada palanca?")
    st.markdown(
        """
**Balance primario máximo factible (BPMF)**  
Es el mayor superávit primario que el Estado podría sostener de manera realista. Si lo subes, estás suponiendo que el país puede ajustarse más cuando la deuda aprieta.

**Intercepto fiscal (α)**  
Es la tendencia de base del balance primario. Si sube, el Estado parte desde una posición fiscal algo más favorable, incluso antes de reaccionar a la deuda.

**Reacción fiscal a la deuda (β)**  
Mide cuánto mejora el balance primario cuando sube la deuda. En términos prácticos: si β es alto, el gobierno corrige con más fuerza frente al deterioro fiscal.

**Probabilidad mínima exigida**  
Define qué tan estricto eres para llamar “prudente” a un nivel de deuda. Si exiges 90%, eres más conservador. Si exiges 80%, aceptas más riesgo.

**Volatilidad del crecimiento**  
Representa qué tan inestable es la economía real. Más volatilidad implica más riesgo de años malos, lo que reduce el límite prudente.

**Volatilidad de la tasa de interés**  
Representa la incertidumbre en el costo de financiamiento. Si sube, hay más riesgo de que la deuda se encarezca de forma brusca.

**Volatilidad fiscal**  
Captura shocks en el balance primario: menor recaudación, más gasto inesperado, crisis, etc. Si sube, la deuda prudente tiende a bajar.

**Otros ajustes**  
Resume elementos que no pasan directamente por el balance primario, como movimientos de caja, capitalizaciones, uso de activos del Tesoro u otras necesidades de financiamiento.

**Nivel de referencia de deuda**  
Es el punto desde el cual el modelo “siente” que la deuda ya está relativamente alta y activa una reacción fiscal más intensa.

**Techo de deuda**  
Es una barrera de seguridad para declarar que una trayectoria dejó de ser razonable. No es una meta política, sino un filtro de riesgo del ejercicio.
"""
    )

with right:
    st.subheader("Lectura práctica")
    st.info(
        """
**Para subir el límite prudente**, normalmente debes asumir una combinación de:
- mejor capacidad de ajuste fiscal,
- menor volatilidad macroeconómica,
- menor costo financiero,
- o una definición menos estricta de prudencia.

**Para bajar el límite prudente**, basta con asumir:
- shocks más frecuentes,
- reacción fiscal más débil,
- mayor tasa de interés,
- o un criterio más exigente de sostenibilidad.
"""
    )

    st.caption(
        "Este modelo es una simplificación pedagógica. Sirve para analizar sensibilidad y lógica macrofiscal, no para reemplazar una estimación oficial."
    )

st.divider()

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Probabilidad de sostenibilidad por deuda inicial")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df_prob["deuda_neta_inicial"] * 100, df_prob["probabilidad_sostenible"] * 100)
    ax.axhline(required_probability * 100, linestyle="--")
    if not np.isnan(prudent):
        ax.axvline(prudent * 100, linestyle=":")
    ax.set_xlabel("Deuda neta inicial (% del PIB)")
    ax.set_ylabel("Probabilidad de sostenibilidad (%)")
    ax.set_title("Frontera de prudencia")
    plt.tight_layout()
    st.pyplot(fig)

with chart2:
    if not np.isnan(prudent):
        st.subheader("Trayectoria simulada usando el nivel prudente")
        prob_ok, p10, p50, p90, pb50 = run_reference_path(d0=prudent, years=years, **params)
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        ax2.plot(years, p50, label="Mediana")
        ax2.fill_between(years, p10, p90, alpha=0.25, label="P10-P90")
        ax2.set_xlabel("Año")
        ax2.set_ylabel("Deuda neta (% del PIB)")
        ax2.set_title("Distribución de trayectorias")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

st.divider()

st.subheader("Tabla de resultados")
show_df = df_prob.copy()
show_df["deuda_neta_inicial"] = show_df["deuda_neta_inicial"] * 100
show_df["probabilidad_sostenible"] = show_df["probabilidad_sostenible"] * 100
show_df.columns = ["Deuda neta inicial (% PIB)", "Probabilidad de sostenibilidad (%)"]
st.dataframe(show_df, use_container_width=True)

csv = show_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar tabla CSV",
    data=csv,
    file_name="resultados_deuda_prudente.csv",
    mime="text/csv",
)

st.divider()

st.subheader("Escenarios rápidos")
quick1, quick2, quick3 = st.columns(3)
quick1.markdown(
    """
**Conservador**  
- BPMF bajo  
- β moderado  
- alta volatilidad  
- alta probabilidad exigida
"""
)
quick2.markdown(
    """
**Intermedio**  
- ajustes razonables  
- shocks moderados  
- criterio de prudencia medio
"""
)
quick3.markdown(
    """
**Menos conservador**  
- mayor capacidad de ajuste  
- menor volatilidad  
- mayor tolerancia al riesgo
"""
)

st.divider()
