import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import random
import re
import google.generativeai as genai

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="ASISTENTE VIRTUAL ESTADISTICA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CONFIGURACIÓN DE GEMINI Y MODELOS
# ---------------------------------------------------------
if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
    API_KEY_GEMINI = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY_GEMINI = ""

if API_KEY_GEMINI:
    genai.configure(api_key=API_KEY_GEMINI)

# SYSTEM PROMPT MINIMALISTA (PROMPT CORTO EVITA PENSAMIENTO EN VOZ ALTA)
SYSTEM_PROMPT_UMG = "Eres el Asistente Virtual de Estadística de la UMG. Responde directamente en español al estudiante de forma clara y didáctica."

MODELOS_RESPALDO = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro"
]

def obtener_lista_modelos():
    """Obtiene la lista de modelos disponibles para la IA."""
    modelos_probar = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                modelos_probar.append(m.name)
    except Exception:
        pass
    
    for m in MODELOS_RESPALDO:
        if m not in modelos_probar:
            modelos_probar.append(m)
            
    return modelos_probar

def sanear_respuesta_ia(texto):
    """Elimina al 100% cualquier borrador o metatexto en inglés previo a la respuesta."""
    if not texto:
        return ""
    
    # 1. Si la IA imprime metatexto antes de responder, recortamos directamente desde ¡Hola! o Hola
    match = re.search(r'(¡?Hola\b.*)', texto, re.DOTALL | re.IGNORECASE)
    if match:
        texto = match.group(1)
        
    # 2. Filtro secundario para eliminar líneas sueltas de metadatos si no hay saludo
    lineas = texto.split("\n")
    lineas_limpias = []
    
    patrones_basura = [
        r'^\s*[-*]\s*(user|role|tone|language|context|constraints?|check|draft|definition|types|example|greeting|closing):',
        r'^\s*(user asks|persona:|constraints:|check rule|draft \d|spanish only|self-correction)',
        r'^\s*[-*]\s*ensure\b',
        r'^\s*[-*]\s*use formatting\b'
    ]
    
    for linea in lineas:
        l_low = linea.lower().strip()
        if any(re.search(p, l_low) for p in patrones_basura):
            continue
        lineas_limpias.append(linea)
        
    return "\n".join(lineas_limpias).strip()

# ---------------------------------------------------------
# FUNCIONES DE IA CON FILTRADO ESTRICTO
# ---------------------------------------------------------
def generar_texto_con_fallback(contents):
    if not API_KEY_GEMINI:
        return "⚠️ No se detectó la clave GEMINI_API_KEY en los Secrets de Streamlit."
    
    ultimo_error = None
    for nombre_modelo in obtener_lista_modelos():
        try:
            model = genai.GenerativeModel(
                nombre_modelo,
                system_instruction=SYSTEM_PROMPT_UMG,
                generation_config={"temperature": 0.1}
            )
            response = model.generate_content(contents)
            if response.text:
                return sanear_respuesta_ia(response.text)
        except Exception as e:
            ultimo_error = e
            continue

    return f"⚠️ Error al conectar con Google Gemini: {ultimo_error}"

def generar_stream_con_fallback(contents):
    if not API_KEY_GEMINI:
        yield "⚠️ No se detectó la clave GEMINI_API_KEY en los Secrets de Streamlit."
        return
    
    ultimo_error = None
    for nombre_modelo in obtener_lista_modelos():
        try:
            model = genai.GenerativeModel(
                nombre_modelo,
                system_instruction=SYSTEM_PROMPT_UMG,
                generation_config={"temperature": 0.1}
            )
            response = model.generate_content(contents)
            if response.text:
                texto_limpio = sanear_respuesta_ia(response.text)
                if texto_limpio:
                    yield texto_limpio
                    return
        except Exception as e:
            ultimo_error = e
            continue

    yield f"⚠️ Error al conectar con Google Gemini: {ultimo_error}"
# ---------------------------------------------------------
# ESTILOS CSS PERSONALIZADOS (SIN BANNER SUPERIOR)
# ---------------------------------------------------------
st.markdown("""
<style>
    div.block-container { padding-top: 2rem !important; } 
    
    .sidebar-header-card { 
        background: linear-gradient(135deg, #183059 0%, #2a5298 100%); 
        padding: 1rem; border-radius: 10px; color: white; text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15); margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .sidebar-facultad {
        color: #fef08a; font-size: 0.72rem; font-weight: 700; 
        text-transform: uppercase; border-bottom: 1px solid rgba(255,255,255,0.2); 
        padding-bottom: 4px; margin-bottom: 6px; letter-spacing: 0.5px;
    }
    .sidebar-title {
        color: #ffffff; font-size: 1.15rem; font-weight: 900; 
        margin: 4px 0 2px 0; text-transform: uppercase; line-height: 1.2;
    }
    .sidebar-sub {
        color: #e0e7ff; font-size: 0.8rem; margin: 0; font-weight: 500;
    }
    .sidebar-sede {
        color: #cbd5e1; font-size: 0.75rem; font-style: italic; margin-bottom: 8px;
    }
    .sidebar-curso {
        background: rgba(255, 255, 255, 0.15); padding: 4px 8px; border-radius: 6px;
        font-size: 0.82rem; font-weight: bold; color: #ffffff; margin: 6px 0;
    }
    .sidebar-integrantes {
        font-size: 0.78rem; text-align: left; background: rgba(0,0,0,0.2);
        padding: 8px; border-radius: 6px; margin-top: 6px; color: #f8fafc;
    }
    
    .sidebar-footer-card {
        background: linear-gradient(135deg, #183059 0%, #2a5298 100%);
        padding: 0.9rem; border-radius: 10px; color: white; text-align: center;
        margin-top: 1.2rem; border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .miumg-cursive-sidebar { 
        font-family: 'Brush Script MT', cursive; font-size: 2.5rem; 
        font-style: italic; font-weight: bold; color: #ffffff; 
        text-shadow: 1px 1px 3px rgba(0,0,0,0.4); line-height: 1; margin-bottom: 4px;
    }
    .lema-linea-sidebar { font-size: 0.75rem; font-style: italic; margin: 0; color: #e0e7ff; }
    .lema-cita-sidebar { font-size: 0.75rem; font-weight: 800; color: #fef08a; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# BARRA LATERAL (ENCABEZADO, INTEGRANTES Y NAVEGACIÓN)
# ---------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-header-card">
    <div class="sidebar-facultad">Facultad de Ingeniería en Sistemas</div>
    <div class="sidebar-title">ASISTENTE VIRTUAL ESTADÍSTICA</div>
    <div class="sidebar-sub">Universidad Mariano Gálvez</div>
    <div class="sidebar-sede">Santa Lucía Cotzumalguapa</div>
    <div class="sidebar-curso">📘 Física II | Sección: A</div>
    <div class="sidebar-integrantes">
        <b>👨‍💻 Integrantes:</b><br>
        • Esdras Morán (<code>2990-19-7648</code>)<br>
        • Emerson Chach (<code>2990-24-2093</code>)
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🧮 Navegación")
opcion = st.sidebar.radio(
    "Selecciona un Módulo:",
    [
        "🏠 Inicio y Guía Rápida",
        "📊 1. Estadística Descriptiva",
        "📈 2. Tablas y Gráficas de Frecuencia",
        "🎲 3. Probabilidad y Normal",
        "📉 4. Regresión Lineal",
        "📚 5. Estudio Guiado"
    ],
    label_visibility="collapsed"
)

# Tarjeta inferior en la barra lateral con la marca UMG y el lema
st.sidebar.markdown("""
<div class="sidebar-footer-card">
    <div class="miumg-cursive-sidebar">MI UMG</div>
    <p class="lema-linea-sidebar">"Y conoceréis la verdad</p>
    <p class="lema-linea-sidebar">y la verdad os hará libres"</p>
    <div class="lema-cita-sidebar">Juan 8:32</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INICIO
# =========================================================
if opcion == "🏠 Inicio y Guía Rápida":
    st.subheader("⚡ Guía Rápida de Uso")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.success("### 1️⃣ Elige un Módulo")
        st.write("Selecciona la herramienta que necesitas en la barra lateral.")
    with col_b:
        st.info("### 2️⃣ Genera o Ingresa Datos")
        st.write("Practica con ejemplos tabulados o resuelve tus propios datos organizados profesionalmente.")
    with col_c:
        st.warning("### 3️⃣ Analiza y Pregunta")
        st.write("Visualiza gráficas 100% traducidas al español, pon a prueba tus conocimientos con quizzes interactivos y resuelve dudas con la IA.")

# =========================================================
# MÓDULO 1: ESTADÍSTICA DESCRIPTIVA
# =========================================================
elif opcion == "📊 1. Estadística Descriptiva":
    st.title("MÓDULO 1: Estadística Descriptiva")
    st.subheader("(Media, mediana, moda, rango y medidas de dispersión)")
    
    st.markdown("### 📚 1. Práctica y Ejemplos Generados")
    subtema1 = st.selectbox(
        "🎯 Selecciona el Subtema a Practicar:",
        [
            "1.1 Media y Mediana",
            "1.2 Varianza y Desviación Estándar Muestral",
            "1.3 Rango y Diagrama de Caja (Boxplot)"
        ],
        key="gen_sub_1"
    )
    
    if st.button("🎲 Generar 10 Ejercicios con Solución y Gráfica", key="btn_gen_1"):
        st.success(f"Ejercicios Generados para: **{subtema1}**")
        for i in range(1, 11):
            datos = sorted([round(random.uniform(5.0, 50.0), 1) for _ in range(random.randint(5, 8))])
            n = len(datos)
            media = np.mean(datos)
            mediana = np.median(datos)
            varianza = np.var(datos, ddof=1)
            desviacion = np.std(datos, ddof=1)
            rango = max(datos) - min(datos)
            
            st.markdown(f"#### 📝 Ejercicio {i} (Muestra n = {n})")
            df_m1 = pd.DataFrame({"Elemento": [f"Dato {j+1}" for j in range(n)], "Valor Observado": datos})
            st.dataframe(df_m1.T, use_container_width=True)
            
            with st.expander(f"👁️ Ver Solución Completa y Gráfica del Ejercicio {i}"):
                if "1.1" in subtema1:
                    st.markdown("**Sustitución y Respuesta:**")
                    st.latex(rf"\bar{{x}} = \frac{{\sum x_i}}{{n}} = \frac{{{sum(datos):.1f}}}{{{n}}} = \mathbf{{{media:.2f}}}")
                    st.latex(rf"\text{{Mediana}} = \mathbf{{{mediana:.2f}}}")
                elif "1.2" in subtema1:
                    st.markdown("**Sustitución y Respuesta:**")
                    st.latex(rf"s^2 = \frac{{\sum (x_i - \bar{{x}})^2}}{{n-1}} = \mathbf{{{varianza:.4f}}}")
                    st.latex(rf"s = \sqrt{{s^2}} = \mathbf{{{desviacion:.4f}}}")
                else:
                    st.markdown("**Sustitución y Respuesta:**")
                    st.latex(rf"\text{{Rango}} = X_{{\max}} - X_{{\min}} = {max(datos)} - {min(datos)} = \mathbf{{{rango:.2f}}}")
                
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=datos,
                    name="Muestra",
                    marker_color='#183059',
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    hovertemplate="<b>Valor:</b> %{y}<extra></extra>"
                ))
                fig.update_layout(title=f"Diagrama de Caja y Dispersión - Ejercicio {i}", yaxis_title="Valores Observados", height=320, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### ✍️ 2. Resolución de tus Propios Ejercicios")
    
    subtema1_user = st.selectbox(
        "🎯 Selecciona el Subtema que deseas calcular:",
        [
            "1.1 Media y Mediana",
            "1.2 Varianza y Desviación Estándar Muestral",
            "1.3 Rango y Diagrama de Caja (Boxplot)"
        ],
        key="user_sub_1"
    )
    
    data_input = st.text_area("⌨️ Ingresa los valores (separados por comas):", placeholder="Ej: 9.1, 18.0, 19.3, 21.2, 28.4, 38.9, 45.6", key="data_m1")
    unit_input = st.text_input("Unidad de Medida (opcional):", placeholder="Ej: Amperios", key="unit_m1")

    if st.button("🚀 Resolver Ejercicio Personalizado", key="btn_user_1"):
        if not data_input.strip():
            st.warning("⚠️ Ingresa valores numéricos.")
        else:
            try:
                data = [float(x.strip()) for x in data_input.split(",") if x.strip() != ""]
                n = len(data)
                if n < 2:
                    st.error("⚠️ Se requieren al menos 2 datos numéricos.")
                else:
                    data_np = np.array(data)
                    media = np.mean(data_np)
                    mediana = np.median(data_np)
                    varianza = np.var(data_np, ddof=1)
                    desviacion = np.std(data_np, ddof=1)
                    rango = max(data) - min(data)
                    unidad = f" {unit_input}" if unit_input.strip() else ""

                    st.markdown("#### 📋 Tabla Resumen de los Datos Ingresados")
                    df_user = pd.DataFrame({"Muestra N°": [f"Dato {j+1}" for j in range(n)], f"Valor ({unit_input if unit_input else 'Unidad'})": data})
                    st.dataframe(df_user.T, use_container_width=True)

                    st.success("✅ Datos procesados correctamente.")
                    st.markdown("#### 🔍 Resultados y Sustitución")

                    if "1.1" in subtema1_user:
                        st.latex(rf"\bar{{x}} = \frac{{\sum x_i}}{{n}} = \frac{{{sum(data):.2f}}}{{{n}}} = \mathbf{{{media:.4f}{unidad}}}")
                        st.latex(rf"\text{{Mediana}} = \mathbf{{{mediana:.4f}{unidad}}}")
                        
                        counts, bins = np.histogram(data, bins=min(n, 6))
                        fig = go.Figure(go.Bar(
                            x=[f"{bins[j]:.1f} - {bins[j+1]:.1f}" for j in range(len(counts))],
                            y=counts,
                            marker_color='#183059',
                            hovertemplate="<b>Rango:</b> %{x}<br><b>Frecuencia:</b> %{y}<extra></extra>"
                        ))
                        fig.update_layout(title="Distribución Frecuencial de tus Datos", xaxis_title=f"Intervalos de Valores ({unit_input})" if unit_input else "Intervalos de Valores", yaxis_title="Frecuencia (Cantidad de datos)", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)

                    elif "1.2" in subtema1_user:
                        st.latex(rf"s^2 = \frac{{\sum (x_i - \bar{{x}})^2}}{{n-1}} = \mathbf{{{varianza:.4f}{unidad}^2}}")
                        st.latex(rf"s = \sqrt{{s^2}} = \mathbf{{{desviacion:.4f}{unidad}}}")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(range(1, n+1)), y=data, mode='markers+lines', name='Datos', marker=dict(size=10, color='#ef4444'), hovertemplate="<b>Punto %{x}:</b> %{y}<extra></extra>"))
                        fig.add_hline(y=media, line_dash="dash", line_color="#183059", annotation_text=f"Media: {media:.2f}")
                        fig.update_layout(title="Dispersión Individual respecto a la Media", xaxis_title="Orden de Observación", yaxis_title=f"Valores ({unit_input})" if unit_input else "Valores", template="plotly_white", height=320)
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.latex(rf"\text{{Rango}} = X_{{\max}} - X_{{\min}} = {max(data)} - {min(data)} = \mathbf{{{rango:.4f}{unidad}}}")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Box(y=data, name="Tus Datos", marker_color='#183059', boxpoints='all', hovertemplate="<b>Valor:</b> %{y}<extra></extra>"))
                        fig.update_layout(title="Diagrama de Caja (Boxplot) de tus Datos", yaxis_title=f"Valores ({unit_input})" if unit_input else "Valores", template="plotly_white", height=350)
                        st.plotly_chart(fig, use_container_width=True)

            except ValueError:
                st.error("❌ Formato numérico incorrecto.")

# =========================================================
# MÓDULO 2: TABLAS Y GRÁFICAS DE FRECUENCIA
# =========================================================
elif opcion == "📈 2. Tablas y Gráficas de Frecuencia":
    st.title("MÓDULO 2: Tablas de Frecuencia")
    st.subheader("(Agrupación de datos, Sturges, Histogramas y Ojiva)")
    
    st.markdown("### 📚 1. Práctica y Ejemplos Generados")
    subtema2 = st.selectbox(
        "🎯 Selecciona el Subtema a Practicar:",
        [
            "2.1 Regla de Sturges (Cálculo de Intervalos K)",
            "2.2 Frecuencias Absolutas, Relativas y Acumuladas",
            "2.3 Histograma de Frecuencias u Ojiva Ascendente"
        ],
        key="gen_sub_2"
    )
    
    if st.button("🎲 Generar 10 Series de Datos con Tabla y Gráfica", key="btn_gen_2"):
        st.success(f"Ejercicios Generados para: **{subtema2}**")
        for i in range(1, 11):
            datos = sorted([random.randint(20, 100) for _ in range(random.randint(15, 25))])
            n = len(datos)
            k = int(1 + 3.322 * np.log10(n))
            counts, bins = np.histogram(datos, bins=k)
            
            df = pd.DataFrame({
                "Intervalo Mín": bins[:-1].round(1),
                "Intervalo Máx": bins[1:].round(1),
                "Frecuencia Absoluta (f)": counts,
                "Frecuencia Acumulada (F)": np.cumsum(counts),
                "Frecuencia Relativa (%)": ((counts / n) * 100).round(2)
            })
            
            st.markdown(f"#### 📝 Ejercicio {i} (Serie de {n} datos)")
            df_raw = pd.DataFrame({"Muestra": [f"D{j+1}" for j in range(n)], "Valor": datos})
            st.dataframe(df_raw.T, use_container_width=True)
            
            with st.expander(f"👁️ Ver Tabla Sturges y Gráfica Interactiva Ejercicio {i}"):
                st.latex(rf"K = 1 + 3.322 \log_{{10}}({n}) \implies K = \mathbf{{{k}\ \text{{clases}}}}")
                st.dataframe(df, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df["Intervalo Máx"], y=df["Frecuencia Absoluta (f)"],
                    name="Histograma (f)", marker_color='#183059',
                    hovertemplate="<b>Límite Superior:</b> %{x}<br><b>Frecuencia:</b> %{y}<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=df["Intervalo Máx"], y=df["Frecuencia Acumulada (F)"],
                    name="Ojiva Ascendente (F)", mode='lines+markers', line=dict(color='#ef4444', width=3),
                    hovertemplate="<b>Límite Superior:</b> %{x}<br><b>Frec. Acumulada:</b> %{y}<extra></extra>"
                ))
                fig.update_layout(title=f"Histograma y Ojiva Ascendente - Ejercicio {i}", height=350, template="plotly_white", xaxis_title="Límite Superior de Clase", yaxis_title="Frecuencia (Cantidad de Datos)")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### ✍️ 2. Resolución de tus Propios Ejercicios")
    
    subtema2_user = st.selectbox(
        "🎯 Selecciona el enfoque de tu tabla:",
        [
            "2.1 Regla de Sturges (Cálculo de Intervalos K)",
            "2.2 Frecuencias Absolutas, Relativas y Acumuladas",
            "2.3 Histograma de Frecuencias u Ojiva Ascendente"
        ],
        key="user_sub_2"
    )
    
    data_input = st.text_area("⌨️ Ingresa la serie de datos (separados por comas):", placeholder="Ej: 10, 12, 14, 15, 11, 13, 18, 20...", key="data_m2")

    if st.button("🚀 Construir Tabla y Gráfica Personalizada", key="btn_user_2"):
        if not data_input.strip():
            st.warning("⚠️ Ingresa los datos numéricos.")
        else:
            try:
                data = [float(x.strip()) for x in data_input.split(",") if x.strip() != ""]
                n = len(data)
                if n < 5:
                    st.error("⚠️ Se sugieren al menos 5 datos para crear una tabla agrupada.")
                else:
                    k = int(1 + 3.322 * np.log10(n))
                    counts, bins = np.histogram(data, bins=k)
                    df = pd.DataFrame({
                        "Clase Mín": bins[:-1].round(2),
                        "Clase Máx": bins[1:].round(2),
                        "Frecuencia Absoluta (f)": counts,
                        "Frecuencia Acumulada (F)": np.cumsum(counts),
                        "Frecuencia Relativa (%)": ((counts / n) * 100).round(2)
                    })
                    
                    st.markdown("#### 📋 Serie de Datos Ingresada")
                    df_user_raw = pd.DataFrame({"N°": [f"D{j+1}" for j in range(n)], "Valor": data})
                    st.dataframe(df_user_raw.T, use_container_width=True)

                    st.success(f"✅ Tabla generada aplicando Sturges con K = {k} intervalos.")
                    st.latex(rf"K = 1 + 3.322 \log_{{10}}({n}) = \mathbf{{{1 + 3.322 * np.log10(n):.2f}}} \approx \mathbf{{{k}\ \text{{intervalos}}}}")
                    st.dataframe(df, use_container_width=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df["Clase Máx"], y=df["Frecuencia Absoluta (f)"],
                        name="Histograma (f)", marker_color='#183059',
                        hovertemplate="<b>Clase Máxima:</b> %{x}<br><b>Frecuencia:</b> %{y}<extra></extra>"
                    ))
                    if "2.3" in subtema2_user:
                        fig.add_trace(go.Scatter(
                            x=df["Clase Máx"], y=df["Frecuencia Acumulada (F)"],
                            name="Ojiva (F)", mode='lines+markers', line=dict(color='#ef4444', width=3),
                            hovertemplate="<b>Clase Máxima:</b> %{x}<br><b>Frec. Acumulada:</b> %{y}<extra></extra>"
                        ))
                    fig.update_layout(title="Histograma de Frecuencia y Ojiva Ascendente", xaxis_title="Intervalos (Límite Superior)", yaxis_title="Frecuencia (Cantidad de datos)", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            except ValueError:
                st.error("❌ Verifique el formato numérico.")

# =========================================================
# MÓDULO 3: PROBABILIDAD Y DISTRIBUCIÓN NORMAL
# =========================================================
elif opcion == "🎲 3. Probabilidad y Normal":
    st.title("MÓDULO 3: Probabilidad y Distribución Normal")
    st.subheader("(Puntuación Z, Área bajo la curva, Campana de Gauss)")
    
    st.markdown("### 📚 1. Práctica y Ejemplos Generados")
    subtema3 = st.selectbox(
        "🎯 Selecciona el Subtema a Practicar:",
        [
            "3.1 Puntuación Z (Estandarización)",
            "3.2 Probabilidad Menor o Igual P(X <= x)",
            "3.3 Probabilidad Mayor o Igual P(X >= x)"
        ],
        key="gen_sub_3"
    )
    
    if st.button("🎲 Generar 10 Ejemplos con Campana de Gauss", key="btn_gen_3"):
        st.success(f"Ejercicios Generados para: **{subtema3}**")
        for i in range(1, 11):
            mu = random.randint(50, 100)
            sig = random.randint(5, 15)
            x_val = mu + random.randint(-12, 12)
            z = (x_val - mu) / sig
            prob_menor = stats.norm.cdf(z)
            prob_mayor = 1 - prob_menor
            
            st.markdown(f"#### 📝 Ejercicio {i}")
            df_params = pd.DataFrame({"Parámetro": ["Media (μ)", "Desviación Estándar (σ)", "Valor Evaluado (X)"], "Valor": [mu, sig, x_val]})
            st.dataframe(df_params.T, use_container_width=True)
            
            with st.expander(f"👁️ Ver Solución y Campana de Gauss Ejercicio {i}"):
                st.latex(rf"Z = \frac{{X - \mu}}{{\sigma}} = \frac{{{x_val} - {mu}}}{{{sig}}} = \mathbf{{{z:.4f}}}")
                
                if "3.2" in subtema3 or "3.1" in subtema3:
                    st.latex(rf"P(X \le {x_val}) = P(Z \le {z:.2f}) = \mathbf{{{prob_menor*100:.2f}\%}}")
                else:
                    st.latex(rf"P(X \ge {x_val}) = 1 - P(Z \le {z:.2f}) = \mathbf{{{prob_mayor*100:.2f}\%}}")
                
                x_axis = np.linspace(mu - 4*sig, mu + 4*sig, 200)
                y_axis = stats.norm.pdf(x_axis, mu, sig)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_axis, y=y_axis, mode='lines', name='Curva Normal',
                    line=dict(color='#183059', width=3),
                    hovertemplate="<b>X:</b> %{x:.2f}<br><b>Densidad:</b> %{y:.4f}<extra></extra>"
                ))
                
                if "3.3" in subtema3:
                    x_shade = np.linspace(x_val, mu + 4*sig, 100)
                else:
                    x_shade = np.linspace(mu - 4*sig, x_val, 100)
                y_shade = stats.norm.pdf(x_shade, mu, sig)
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[x_shade[0]], x_shade, [x_shade[-1]]]), 
                    y=np.concatenate([[0], y_shade, [0]]), 
                    fill='tozeroy', fillcolor='rgba(43, 82, 152, 0.4)', name='Área de Probabilidad',
                    hovertemplate="<b>Zona Evaluada</b><extra></extra>"
                ))
                
                fig.update_layout(title=f"Campana de Gauss - Ejercicio {i}", height=320, template="plotly_white", xaxis_title="Valores Observados (X)", yaxis_title="Densidad de Probabilidad")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### ✍️ 2. Resolución de tus Propios Ejercicios")
    
    subtema3_user = st.selectbox(
        "🎯 Selecciona el Subtema a Evaluar:",
        [
            "3.1 Puntuación Z (Estandarización)",
            "3.2 Probabilidad Menor o Igual P(X <= x)",
            "3.3 Probabilidad Mayor o Igual P(X >= x)"
        ],
        key="user_sub_3"
    )
    
    c1, c2, c3 = st.columns(3)
    with c1: 
        mu_in = st.text_input("Media (μ):", placeholder="Ej: 50", key="mu_m3")
    with c2: 
        sig_in = st.text_input("Desviación (σ):", placeholder="Ej: 5", key="sig_m3")
    with c3: 
        x_in = st.text_input("Valor X a evaluar:", placeholder="Ej: 55", key="x_m3")

    if st.button("🚀 Calcular Probabilidad y Graficar", key="btn_user_3"):
        if not mu_in or not sig_in or not x_in:
            st.warning("⚠️ Ingresa μ, σ y X.")
        else:
            try:
                mu, sigma, x_val = float(mu_in), float(sig_in), float(x_in)
                z = (x_val - mu) / sigma
                prob_menor = stats.norm.cdf(x_val, mu, sigma)
                prob_mayor = 1.0 - prob_menor
                
                st.markdown("#### 📋 Parámetros de Entrada")
                df_user_p = pd.DataFrame({"Variable": ["Media (μ)", "Desviación (σ)", "Punto X"], "Valor": [mu, sigma, x_val]})
                st.dataframe(df_user_p.T, use_container_width=True)

                st.success("✅ Cálculo Realizado.")
                st.latex(rf"Z = \frac{{{x_val} - {mu}}}{{{sigma}}} = \mathbf{{{z:.4f}}}")
                
                if "3.3" in subtema3_user:
                    st.write(rf"**Probabilidad $P(X \ge {x_val})$:** $\mathbf{{{prob_mayor*100:.2f}\%}}$")
                else:
                    st.write(rf"**Probabilidad $P(X \le {x_val})$:** $\mathbf{{{prob_menor*100:.2f}\%}}$")
                
                x_axis = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
                y_axis = stats.norm.pdf(x_axis, mu, sigma)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_axis, y=y_axis, mode='lines', name='Curva Normal',
                    line=dict(color='#183059', width=3),
                    hovertemplate="<b>X:</b> %{x:.2f}<br><b>Densidad:</b> %{y:.4f}<extra></extra>"
                ))
                
                if "3.3" in subtema3_user:
                    x_shade = np.linspace(x_val, mu + 4*sigma, 100)
                else:
                    x_shade = np.linspace(mu - 4*sigma, x_val, 100)
                y_shade = stats.norm.pdf(x_shade, mu, sigma)
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[x_shade[0]], x_shade, [x_shade[-1]]]), 
                    y=np.concatenate([[0], y_shade, [0]]), 
                    fill='tozeroy', fillcolor='rgba(43, 82, 152, 0.4)', name='Área de Probabilidad',
                    hovertemplate="<b>Zona Evaluada</b><extra></extra>"
                ))
                
                fig.update_layout(title="Tu Campana de Gauss Personalizada", height=350, template="plotly_white", xaxis_title="Valores Observados (X)", yaxis_title="Densidad de Probabilidad")
                st.plotly_chart(fig, use_container_width=True)

            except ValueError:
                st.error("❌ Solo se permiten valores numéricos.")

# =========================================================
# MÓDULO 4: REGRESIÓN LINEAL
# =========================================================
elif opcion == "📉 4. Regresión Lineal":
    st.title("MÓDULO 4: Regresión Lineal y Correlación")
    st.subheader("(Recta de Ajuste, Pearson r y Coeficiente R²)")
    
    st.markdown("### 📚 1. Práctica y Ejemplos Generados")
    subtema4 = st.selectbox(
        "🎯 Selecciona el Subtema a Practicar:",
        [
            "4.1 Ecuación de la Recta de Ajuste (Y = a + bX)",
            "4.2 Coeficiente de Correlación de Pearson (r)",
            "4.3 Coeficiente de Determinación (R²)"
        ],
        key="gen_sub_4"
    )
    
    if st.button("🎲 Generar 10 Ejemplos con Diagrama de Dispersión", key="btn_gen_4"):
        st.success(f"Ejercicios Generados para: **{subtema4}**")
        for i in range(1, 11):
            n_puntos = 6
            x_vals = sorted([random.randint(1, 10) for _ in range(n_puntos)])
            m_rand = round(random.uniform(1.2, 3.0), 1)
            b_rand = round(random.uniform(2.0, 5.0), 1)
            y_vals = [round(m_rand * x + b_rand + random.uniform(-1.2, 1.2), 1) for x in x_vals]
            
            slope, inter, r_val, p_val, std_err = stats.linregress(x_vals, y_vals)
            
            st.markdown(f"#### 📝 Ejercicio {i}")
            df_reg = pd.DataFrame({"Punto": [f"P{j+1}" for j in range(n_puntos)], "Variable X": x_vals, "Variable Y": y_vals})
            st.dataframe(df_reg.T, use_container_width=True)
            
            with st.expander(f"👁️ Ver Solución y Recta de Regresión Ejercicio {i}"):
                st.latex(rf"\hat{{Y}} = \mathbf{{{inter:.4f}}} + \mathbf{{{slope:.4f}}}X")
                st.latex(rf"r = \mathbf{{{r_val:.4f}}} \quad | \quad R^2 = \mathbf{{{(r_val**2):.4f}}}")
                
                x_trend = np.array([min(x_vals), max(x_vals)])
                y_trend = inter + slope * x_trend
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='markers', name='Puntos (X,Y)',
                    marker=dict(size=10, color='#ef4444'),
                    hovertemplate="<b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=x_trend, y=y_trend, mode='lines', name='Recta de Ajuste',
                    line=dict(color='#183059', width=3),
                    hovertemplate="<b>Recta Calculada</b><extra></extra>"
                ))
                fig.update_layout(title=f"Ajuste Lineal - Ejercicio {i}", height=320, template="plotly_white", xaxis_title="Variable Independiente (X)", yaxis_title="Variable Dependiente (Y)")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### ✍️ 2. Resolución de tus Propios Ejercicios")
    
    subtema4_user = st.selectbox(
        "🎯 Selecciona la métrica que deseas destacar:",
        [
            "4.1 Ecuación de la Recta de Ajuste (Y = a + bX)",
            "4.2 Coeficiente de Correlación de Pearson (r)",
            "4.3 Coeficiente de Determinación (R²)"
        ],
        key="user_sub_4"
    )
    
    cx, cy = st.columns(2)
    with cx: 
        x_str = st.text_input("Valores X (separados por comas):", placeholder="Ej: 1, 2, 3, 4, 5", key="x_m4")
    with cy: 
        y_str = st.text_input("Valores Y (separados por comas):", placeholder="Ej: 2.1, 3.9, 6.1, 8.2, 9.8", key="y_m4")

    if st.button("🚀 Calcular Regresión y Graficar", key="btn_user_4"):
        if not x_str or not y_str:
            st.warning("⚠️ Ingresa los datos numéricos.")
        else:
            try:
                x_vals = np.array([float(v.strip()) for v in x_str.split(",")])
                y_vals = np.array([float(v.strip()) for v in y_str.split(",")])
                if len(x_vals) != len(y_vals) or len(x_vals) < 2:
                    st.error("⚠️ X e Y deben tener la misma cantidad de datos (mínimo 2).")
                else:
                    slope, inter, r_val, p_val, std_err = stats.linregress(x_vals, y_vals)
                    
                    st.markdown("#### 📋 Pares Numéricos Ingresados")
                    df_user_reg = pd.DataFrame({"Punto N°": [f"P{j+1}" for j in range(len(x_vals))], "X": x_vals, "Y": y_vals})
                    st.dataframe(df_user_reg.T, use_container_width=True)

                    st.success("✅ Ajuste Realizado.")
                    
                    if "4.1" in subtema4_user:
                        st.latex(rf"\hat{{Y}} = \mathbf{{{inter:.4f}}} + \mathbf{{{slope:.4f}}}X")
                    elif "4.2" in subtema4_user:
                        st.write(rf"**Coeficiente de Correlación de Pearson ($r$):** $\mathbf{{{r_val:.4f}}}$")
                    else:
                        st.write(rf"**Coeficiente de Determinación ($R^2$):** $\mathbf{{{(r_val**2):.4f}}}$")

                    x_trend = np.array([min(x_vals), max(x_vals)])
                    y_trend = inter + slope * x_trend
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode='markers', name='Tus Datos',
                        marker=dict(size=10, color='#ef4444'),
                        hovertemplate="<b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>"
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_trend, y=y_trend, mode='lines', name='Recta Ajustada',
                        line=dict(color='#183059', width=3),
                        hovertemplate="<b>Recta de Ajuste</b><extra></extra>"
                    ))
                    fig.update_layout(title="Tu Diagrama de Dispersión y Recta de Ajuste", height=350, template="plotly_white", xaxis_title="Variable Independiente (X)", yaxis_title="Variable Dependiente (Y)")
                    st.plotly_chart(fig, use_container_width=True)
            except ValueError:
                st.error("❌ Formato numérico incorrecto.")

# =========================================================
# MÓDULO 5: ESTUDIO GUIADO, LECCIONES, QUIZ E IA
# =========================================================
elif opcion == "📚 5. Estudio Guiado":
    st.title("📚 Módulo de Estudio Guiado y Asistente IA")
    st.write("Selecciona un subtema para estudiar su teoría, analizar ejemplos prácticos, resolver un Quiz evaluativo y realizar consultas a la IA.")

    # ---------------------------------------------------------
    # ESTRUCTURA DE LECCIONES TEÓRICAS Y FÓRMULAS LATEX
    # ---------------------------------------------------------
    TEORIA_SUBTEMAS = {
        "1.1 Medidas de Tendencia Central (Media, Mediana, Moda)": {
            "historia": "El concepto de media aritmética se remonta a la antigua Grecia con Pitágoras y Arquímedes. En el siglo XIX, Adolphe Quetelet aplicó estas medidas para caracterizar el valor representativo central de un conjunto de mediciones físicas y experimentales.",
            "teoria": "Las medidas de tendencia central ubican el centro de gravedad de una distribución de datos. La Media es el promedio ponderado, la Mediana separa la muestra ordenada exactamente al 50%, y la Moda representa el dato con mayor frecuencia.",
            "formulas": [r"\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}", r"\text{Mediana} = x_{\left(\frac{n+1}{2}\right)}"]
        },
        "1.2 Medidas de Dispersión (Varianza, Desviación Estándar)": {
            "historia": "Karl Pearson introdujo formalmente el término 'desviación estándar' en 1893 para cuantificar la variabilidad en mediciones físicas, superando las limitaciones teóricas de la desviación media.",
            "teoria": "Indican qué tan dispersos están los datos respecto a la media. Una desviación estándar baja indica que los datos están agrupados cerca del promedio; una alta indica alta variabilidad experimental.",
            "formulas": [r"s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}", r"s = \sqrt{s^2}"]
        },
        "1.3 Error Experimental y Manejo de Unidades en Física": {
            "historia": "Carl Friedrich Gauss desarrolló la teoría del error experimental para corregir pequeñas imprecisiones en observaciones astronómicas y físicas.",
            "teoria": "En Física, ningún instrumento proporciona una medida infinitamente exacta. Toda medición conlleva un margen de error absoluto y relativo que debe cuantificarse.",
            "formulas": [r"\text{Error Absoluto} = |V_{\text{medido}} - V_{\text{real}}|", r"\text{Error Relativo \%} = \frac{|V_{\text{medido}} - V_{\text{real}}|}{V_{\text{real}}} \times 100"]
        },
        "2.1 Regla de Sturges y Construcción de Intervalos": {
            "historia": "Formulada por Herbert Sturges en 1926 mediante la expansión de la distribución binomial para estimar el número idóneo de clases en histogramas.",
            "teoria": "Permite agrupar una masa de datos numéricos continuos en intervalos de clase óptimos sin perder la estructura representativa de la población.",
            "formulas": [r"K = 1 + 3.322 \log_{10}(n)", r"W = \frac{X_{\max} - X_{\min}}{K}"]
        },
        "2.2 Frecuencia Absoluta, Relativa y Acumulada": {
            "historia": "Las tablas de frecuencias surgieron de los censos de población en el siglo XVII liderados por John Graunt para resumir grandes cantidades de observaciones.",
            "teoria": "La Frecuencia Absoluta es el conteo simple; la Relativa indica la proporción respecto al total; y la Acumulada suma progresivamente las frecuencias absolutas.",
            "formulas": [r"f_r = \frac{f_i}{n}", r"F_i = \sum_{j=1}^{i} f_j"]
        },
        "2.3 Histograma de Frecuencias y Ojiva Ascendente": {
            "historia": "Karl Pearson introdujo el término 'histograma' en 1891 para visualizar gráficos de barras contiguas asociadas a variables cuantitativas continuas.",
            "teoria": "El histograma describe la forma de la distribución (sesgo o simetría) mientras que la ojiva grafica el incremento progresivo de la frecuencia acumulada.",
            "formulas": [r"\text{Eje X: Límites de Clase}", r"\text{Eje Y: Frecuencia Absoluta } f_i \text{ u Ojiva } F_i"]
        },
        "3.1 Propiedades de la Campana de Gauss (N(μ, σ))": {
            "historia": "Estudiada por Abraham de Moivre en 1733 y desarrollada plenamente por Carl Friedrich Gauss en 1809 para el análisis de distribuciones continuas.",
            "teoria": "Es una curva simétrica e unimodal en forma de campana cuyo área total encerrada bajo la curva equivale al 100% de la probabilidad.",
            "formulas": [r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}"]
        },
        "3.2 Puntuación Z y Estandarización de Variables": {
            "historia": "La estandarización unificó el uso de tablas universales para cualquier distribución normal N(μ, σ) al convertirla a la forma estándar N(0, 1).",
            "teoria": "La puntuación Z mide a cuántas desviaciones estándar se ubica una observación concreta X por encima o por debajo de la media poblacional μ.",
            "formulas": [r"Z = \frac{X - \mu}{\sigma}"]
        },
        "3.3 Cálculo de Probabilidades Acumuladas P(X ≤ x)": {
            "historia": "Permitió calcular áreas bajo la curva normal estándar reduciendo la integración matemática compleja al uso de tablas tipificadas acumuladas.",
            "teoria": "Representa la proporción acumulada de la población cuya variable es menor o igual a un determinado límite x.",
            "formulas": [r"P(X \le x) = P\left(Z \le \frac{x - \mu}{\sigma}\right) = \Phi(Z)"]
        },
        "4.1 Diagrama de Dispersión y Recta de Ajuste": {
            "historia": "Sir Francis Galton derivó el concepto de regresión lineal en el siglo XIX al estudiar la herencia genética de estaturas entre padres e hijos.",
            "teoria": "Modela numéricamente la tendencia lineal entre una variable independiente X y una dependiente Y mediante el método de mínimos cuadrados ordinarios.",
            "formulas": [r"\hat{Y} = a + bX", r"b = \frac{n\sum XY - \sum X \sum Y}{n\sum X^2 - (\sum X)^2}"]
        },
        "4.2 Coeficiente de Correlación de Pearson (r)": {
            "historia": "Desarrollado por Karl Pearson a finales del siglo XIX como la medida estándar para graduar la intensidad de relaciones lineales.",
            "teoria": "Toma valores entre -1 y +1. Cerca de +1 indica relación positiva fuerte; cerca de -1 indica relación negativa; cerca de 0 indica ausencia de asociación lineal.",
            "formulas": [r"r = \frac{n\sum XY - \sum X \sum Y}{\sqrt{[n\sum X^2 - (\sum X)^2][n\sum Y^2 - (\sum Y)^2]}}"]
        },
        "4.3 Coeficiente de Determinación (R²)": {
            "historia": "Surgió del trabajo de Sewall Wright en 1921 para determinar la proporción de variabilidad explicada por variables explicativas.",
            "teoria": "Indica qué porcentaje de la variabilidad total de la variable dependiente Y es explicado por la recta de regresión ajustada.",
            "formulas": [r"R^2 = (r)^2 \times 100\%"]
        }
    }

    # ---------------------------------------------------------
    # QUIZZES DE PREGUNTAS POR SUBTEMA
    # ---------------------------------------------------------
    QUIZZES = {
        "1.1 Medidas de Tendencia Central (Media, Mediana, Moda)": [
            {"p": "¿Qué medida de tendencia central es más sensible a valores extremos o atípicos?", "o": ["Mediana", "Media", "Moda"], "r": 1, "exp": "La media utiliza la suma de todos los valores; por ende, un número atípico alto o bajo la altera sustancialmente."},
            {"p": "Si ordenamos la muestra [5, 8, 12, 15, 20], ¿cuál es el valor de la mediana?", "o": ["8", "12", "15"], "r": 1, "exp": "En una muestra ordenada de 5 datos, el valor central en la posición 3 es 12."},
            {"p": "Dada la muestra de frecuencias [2, 3, 3, 5, 7, 3, 9], ¿cuál es la moda?", "o": ["3", "5", "9"], "r": 0, "exp": "El número 3 se repite 3 veces, siendo la mayor frecuencia absoluta."},
            {"p": "Ejercicio Práctico: Dada la muestra [10, 15, 20, 25, 30], calcule la Media (x̄).", "o": ["18.0", "20.0", "22.5"], "r": 1, "exp": "x̄ = (10 + 15 + 20 + 25 + 30) / 5 = 100 / 5 = 20.0."},
            {"p": "¿En qué tipo de distribución coinciden exactamente la Media, Mediana y Moda?", "o": ["Distribución Asimétrica Positiva", "Distribución Simétrica Unimodal", "Distribución Bimodal"], "r": 1, "exp": "En una distribución simétrica perfecta e unimodal, los tres indicadores se cruzan en el centro exacto."}
        ],
        "1.2 Medidas de Dispersión (Varianza, Desviación Estándar)": [
            {"p": "¿En qué unidades de medida se expresa la Desviación Estándar (s)?", "o": ["Unidades al cuadrado", "En las mismas unidades de los datos originales", "Es una magnitud adimensional"], "r": 1, "exp": "Al extraer la raíz cuadrada a la varianza, se recupera la unidad física original."},
            {"p": "Si todos los elementos de una muestra son idénticos a 15, ¿cuánto vale la varianza?", "o": ["15", "1", "0"], "r": 2, "exp": "Si no hay variación entre los datos y la media, la dispersión es totalmente nula (0)."},
            {"p": "Ejercicio Práctico: Para los datos [2, 4, 6] con media x̄ = 4, calcule la Varianza Muestral (s²).", "o": ["2.00", "4.00", "8.00"], "r": 1, "exp": "s² = [(2-4)² + (4-4)² + (6-4)²] / (3-1) = [4 + 0 + 4] / 2 = 8 / 2 = 4.00."},
            {"p": "Ejercicio Práctico: Si la varianza muestral es s² = 16, ¿cuál es la desviación estándar (s)?", "o": ["2.00", "4.00", "8.00"], "r": 1, "exp": "s = √16 = 4.00."},
            {"p": "¿Por qué se divide entre (n - 1) y no entre (n) al calcular la varianza muestral?", "o": ["Para corregir el sesgo de estimación poblacional", "Por convención matemática aleatoria", "Para aumentar el tamaño de la muestra"], "r": 0, "exp": "El divisor (n - 1) proporciona un estimador insesgado de la varianza poblacional."}
        ],
        "1.3 Error Experimental y Manejo de Unidades en Física": [
            {"p": "¿Cómo se define formalmente el Error Absoluto en una medición física?", "o": ["Diferencia absoluta entre valor medido y valor real", "Proporción porcentual del instrumento", "Incertidumbre del fabricante dividida entre dos"], "r": 0, "exp": "Error Absoluto = |Valor Medido - Valor Real|."},
            {"p": "Ejercicio Práctico: Si el valor teórico es 50.0 cm y el medido es 51.0 cm, ¿cuál es el Error Relativo Porcentual?", "o": ["1.0%", "2.0%", "5.0%"], "r": 1, "exp": "Error Relativo % = (|51 - 50| / 50) * 100 = (1 / 50) * 100 = 2.0%."},
            {"p": "Ejercicio Práctico: Se mide un voltaje con un valor real de 200 V y el voltímetro registra 190 V. ¿Cuál es el Error Absoluto?", "o": ["5 V", "10 V", "15 V"], "r": 1, "exp": "Error Absoluto = |190 - 200| = 10 V."},
            {"p": "¿A qué se deben principalmente los Errores Sistemáticos?", "o": ["Variaciones ambientales térmicas aleatorias", "Descalibración del instrumento o fallas en el método", "Errores al azar del observador humano"], "r": 1, "exp": "Los errores sistemáticos son constantes y repetibles, originados por fallas en equipos o métodos."},
            {"p": "¿Qué cualidad física indica la cercanía de varias mediciones entre sí?", "o": ["Exactitud", "Precisión", "Sensibilidad"], "r": 1, "exp": "La precisión mide la repetibilidad y concordancia interna entre mediciones sucesivas."}
        ],
        "2.1 Regla de Sturges y Construcción de Intervalos": [
            {"p": "Ejercicio Práctico: Para n = 100 datos, ¿cuántas clases K recomienda la Regla de Sturges?", "o": ["6 clases", "8 clases", "12 clases"], "r": 1, "exp": "K = 1 + 3.322 * log10(100) = 1 + 3.322(2) = 7.644 ≈ 8 clases."},
            {"p": "Ejercicio Práctico: Si el Rango es 35 y el número de clases es K = 5, ¿cuál es el Ancho de Clase (W)?", "o": ["5", "7", "35"], "r": 1, "exp": "W = Rango / K = 35 / 5 = 7."},
            {"p": "¿Cuál es el objetivo principal de aplicar la Regla de Sturges?", "o": ["Determinar la media poblacional", "Establecer el número adecuado de intervalos sin perder la forma de la muestra", "Calcular el error experimental"], "r": 1, "exp": "Sturges define un número óptimo de clases para agrupar datos de forma simétrica e imparcial."},
            {"p": "Ejercicio Práctico: Si Xmax = 90 y Xmin = 10, ¿cuál es el Rango total de la serie?", "o": ["80", "100", "45"], "r": 0, "exp": "Rango = Xmax - Xmin = 90 - 10 = 80."},
            {"p": "¿Qué sucede si se eligen demasiadas clases (K muy elevado) en un histograma?", "o": ["Se pierde la visión general y aparecen clases vacías", "La muestra se vuelve perfectamente normal", "El histograma se transforma en ojiva"], "r": 0, "exp": "Demasiadas clases fragmentan la información y generan vacíos que dificultan identificar tendencias."}
        ],
        "2.2 Frecuencia Absoluta, Relativa y Acumulada": [
            {"p": "¿A cuánto debe ser igual la suma de todas las frecuencias relativas en una tabla?", "o": ["Al total de datos (n)", "A 1.0 (o 100%)", "Al número de clases (K)"], "r": 1, "exp": "La suma de las proporciones relativas siempre completa el total unitario (1.0 o 100%)."},
            {"p": "Ejercicio Práctico: Si f_i = 8 en una clase y la muestra total es n = 40, ¿cuál es la Frecuencia Relativa (%)?", "o": ["10%", "20%", "25%"], "r": 1, "exp": "f_r = (8 / 40) * 100 = 0.20 * 100 = 20%."},
            {"p": "Ejercicio Práctico: Si las frecuencias absolutas son [3, 5, 2, 4], ¿cuál es la Frecuencia Acumulada (F) de la 3ra clase?", "o": ["8", "10", "14"], "r": 1, "exp": "F3 = f1 + f2 + f3 = 3 + 5 + 2 = 10."},
            {"p": "¿A qué valor corresponde siempre la Frecuencia Acumulada de la última clase?", "o": ["Al Rango", "Al número total de observaciones (n)", "Al Ancho de clase"], "r": 1, "exp": "La Frecuencia Acumulada suma todos los elementos hasta incluir el 100% de la muestra n."},
            {"p": "¿Cómo se calcula la Marca de Clase (X_m)?", "o": ["(Límite Inferior + Límite Superior) / 2", "Límite Superior - Límite Inferior", "Frecuencia Absoluta / n"], "r": 0, "exp": "La Marca de Clase es el punto medio del intervalo."}
        ],
        "2.3 Histograma de Frecuencias y Ojiva Ascendente": [
            {"p": "¿Qué tipo de frecuencia se ubica en el eje vertical (Y) para construir una Ojiva Ascendente?", "o": ["Frecuencia Absoluta (f)", "Frecuencia Acumulada (F)", "Frecuencia Relativa Simple"], "r": 1, "exp": "La Ojiva grafica el crecimiento acumulativo mediante las frecuencias acumuladas (F)."},
            {"p": "A diferencia de un diagrama de barras categórico, ¿por qué en el histograma las barras van unidas?", "o": ["Por razones estéticas", "Porque representan una variable cuantitativa continua", "Para ahorrar espacio gráfico"], "r": 1, "exp": "Las barras contiguas reflejan la continuidad de los intervalos reales de clase."},
            {"p": "Ejercicio Práctico: Una ojiva finaliza su trazado en la última clase alcanzando el valor Y = 50. ¿Cuántos datos se evaluaron?", "o": ["25 datos", "50 datos", "100 datos"], "r": 1, "exp": "El tope de la ojiva ascendente coincide siempre con el tamaño total de la muestra (n = 50)."},
            {"p": "¿Qué punto del intervalo se utiliza en el eje X para trazar el Polígono de Frecuencias?", "o": ["Límite Inferior", "Límite Superior", "Marca de Clase (Punto Medio)"], "r": 2, "exp": "El polígono une los puntos superiores ubicados en la Marca de Clase de cada intervalo."},
            {"p": "¿Qué nos indica la pendiente más empinada en el trazado de una Ojiva?", "o": ["El intervalo con menor frecuencia", "El intervalo con mayor concentración o frecuencia de datos", "Que la muestra es asimétrica negativa"], "r": 1, "exp": "Una pendiente fuerte refleja un incremento rápido de datos acumulados en ese rango."}
        ],
        "3.1 Propiedades de la Campana de Gauss (N(μ, σ))": [
            {"p": "Según la Regla Empírica, ¿qué porcentaje de datos cae aproximadamente en el rango (μ - 1σ, μ + 1σ)?", "o": ["50.0%", "68.3%", "95.4%"], "r": 1, "exp": "La curva normal encierra aproximadamente el 68.3% del área total en ±1 desviación estándar."},
            {"p": "¿Cuánto equivale el área total encerrada debajo de la curva normal de probabilidad?", "o": ["0.5", "1.0 (o 100%)", "2.0"], "r": 1, "exp": "El área bajo toda la curva normal representa el espacio muestral completo (1.0)."},
            {"p": "Ejercicio Práctico: Si una distribución normal tiene μ = 100 y σ = 15, ¿cuál es el rango para el 68.3% central?", "o": ["[85, 115]", "[70, 130]", "[55, 145]"], "r": 0, "exp": "Rango = [100 - 15, 100 + 15] = [85, 115]."},
            {"p": "¿En qué punto exacto alcanza la curva normal su máxima altura o densidad de probabilidad?", "o": ["En X = 0 siempre", "En X = μ (la media)", "En X = σ"], "r": 1, "exp": "La cúspide de la campana se localiza en la media aritmética μ."},
            {"p": "¿Qué caracteriza a las 'colas' de la distribución normal con respecto al eje horizontal?", "o": ["Son asintóticas (se acercan a cero pero nunca lo tocan)", "Cortan al eje en ±3σ", "Tienen valor negativo"], "r": 0, "exp": "Las colas se extienden indefinidamente en dirección a -∞ y +∞ de forma asintótica."}
        ],
        "3.2 Puntuación Z y Estandarización de Variables": [
            {"p": "Ejercicio Práctico: Si μ = 50, σ = 10 y un valor observado es X = 65, ¿cuál es su Puntuación Z?", "o": ["1.00", "1.50", "2.00"], "r": 1, "exp": "Z = (65 - 50) / 10 = 15 / 10 = 1.50."},
            {"p": "Ejercicio Práctico: Si un valor X es exactamente igual a la media (X = μ), ¿cuánto vale Z?", "o": ["0.00", "1.00", "-1.00"], "r": 0, "exp": "Z = (μ - μ) / σ = 0 / σ = 0.00."},
            {"p": "Ejercicio Práctico: Para μ = 100, σ = 20 y X = 80, calcule Z.", "o": ["-1.00", "1.00", "-2.00"], "r": 0, "exp": "Z = (80 - 100) / 20 = -20 / 20 = -1.00."},
            {"p": "¿Qué significa que una puntuación Z resulte negativa?", "o": ["Que se cometió un error de cálculo", "Que el valor X observado se ubica por debajo de la media μ", "Que la desviación estándar es negativa"], "r": 1, "exp": "Un valor Z negativo indica la posición del dato a la izquierda de la media."},
            {"p": "¿Cuáles son los parámetros de la Distribución Normal Estandarizada Z?", "o": ["μ = 1, σ = 0", "μ = 0, σ = 1", "μ = 100, σ = 15"], "r": 1, "exp": "La estandarización normaliza cualquier distribución a media 0 y desviación estándar 1."}
        ],
        "3.3 Cálculo de Probabilidades Acumuladas P(X ≤ x)": [
            {"p": "Ejercicio Práctico: Si P(Z ≤ 1.0) = 0.8413, ¿cuál es la probabilidad mayor P(Z > 1.0)?", "o": ["0.8413", "0.1587", "0.5000"], "r": 1, "exp": "P(Z > 1.0) = 1 - P(Z ≤ 1.0) = 1 - 0.8413 = 0.1587 (15.87%)."},
            {"p": "Ejercicio Práctico: En una normal estándar, ¿cuánto vale P(Z ≤ 0.0)?", "o": ["0.0000", "0.5000", "1.0000"], "r": 1, "exp": "Por la simetría de la campana, el área acumulada hasta la media Z = 0 es exactamente la mitad (0.5000 o 50%)."},
            {"p": "Ejercicio Práctico: Si la masa tiene μ = 70 kg y σ = 10 kg, ¿cuál es la P(X ≤ 70 kg)?", "o": ["25%", "50%", "75%"], "r": 1, "exp": "Z = (70 - 70) / 10 = 0. Como Z = 0, P(X ≤ 70) = 50%."},
            {"p": "¿Cómo se obtiene la probabilidad en un intervalo entre dos puntos P(a ≤ X ≤ b)?", "o": ["P(Z ≤ b) - P(Z ≤ a)", "P(Z ≤ b) + P(Z ≤ a)", "P(Z ≤ b) / P(Z ≤ a)"], "r": 0, "exp": "Se resta la probabilidad acumulada del límite inferior a la del límite superior."},
            {"p": "¿Qué representa la función de distribución acumulada Φ(Z)?", "o": ["La probabilidad de obtener un valor exactamente igual a Z", "El área acumulada bajo la curva desde -∞ hasta Z", "La altura máxima de la curva normal"], "r": 1, "exp": "Φ(Z) representa la probabilidad acumulada P(Z ≤ z)."}
        ],
        "4.1 Diagrama de Dispersión y Recta de Ajuste": [
            {"p": "En la ecuación de regresión Y = a + bX, ¿qué representa el parámetro 'b'?", "o": ["El punto de intersección con el eje Y", "La pendiente o tasa de cambio de Y por unidad de X", "El coeficiente de correlación"], "r": 1, "exp": "La pendiente 'b' mide cuánto varía Y ante un cambio unitario en X."},
            {"p": "Ejercicio Práctico: Si la recta ajustada es Y = 3.0 + 2.0X, ¿cuál es el valor estimado de Y para X = 4.0?", "o": ["9.0", "11.0", "14.0"], "r": 1, "exp": "Y = 3.0 + 2.0(4.0) = 3.0 + 8.0 = 11.0."},
            {"p": "¿Bajo qué criterio matemático se calcula la mejor recta de ajuste en Regresión Lineal?", "o": ["Minimizar la suma de residuos al cuadrado (Mínimos Cuadrados Ordinarios)", "Maximizar las distancias horizontales de los puntos", "Promediar las pendientes de todos los pares"], "r": 0, "exp": "El método MCO minimiza ∑(Y - Y_estimado)²."},
            {"p": "Ejercicio Práctico: Si a = 5.0 y b = 0.0, ¿qué tipo de relación existe entre X e Y?", "o": ["Relación lineal positiva perfecta", "Y es constante e independiente de X (línea horizontal)", "Relación lineal negativa perfecta"], "r": 1, "exp": "Si la pendiente b = 0, Y no cambia independientemente del valor que tome X."},
            {"p": "¿Qué es un 'Residuo' o 'Error de Estimación' (e_i) en regresión?", "o": ["La diferencia entre el valor real Y_i y el valor predicho Y_estimado", "El valor del intercepto a", "La correlación de Pearson al cuadrado"], "r": 0, "exp": "Residuo = Y_observado - Y_predicho por la recta."}
        ],
        "4.2 Coeficiente de Correlación de Pearson (r)": [
            {"p": "¿En qué rango delimitado de valores se encuentra el coeficiente r de Pearson?", "o": ["De 0.0 a 1.0", "De -1.0 a +1.0", "De -100 a +100"], "r": 1, "exp": "El coeficiente r está acotado en el intervalo [-1.0, +1.0]."},
            {"p": "Ejercicio Práctico: Se calcula r = -0.92 para dos variables físicas. ¿Cómo se interpreta?", "o": ["Fuerte asociación lineal positiva", "Débil asociación lineal sin importancia", "Fuerte asociación lineal negativa"], "r": 2, "exp": "Un valor cercano a -1.0 indica que al aumentar X, Y disminuye de forma lineal intensa."},
            {"p": "Si r = 0.00 entre dos variables, ¿cuál es la conclusión adecuada?", "o": ["No existe ninguna relación de ningún tipo entre las variables", "No existe asociación lineal entre las variables", "Los datos forman una recta perfecta"], "r": 1, "exp": "Un r = 0 descarta asociación lineal, pero podrían existir relaciones no lineales (curvilíneas)."},
            {"p": "Ejercicio Práctico: Si al aumentar la temperatura X, la resistencia Y aumenta formando una línea recta perfecta, ¿cuánto vale r?", "o": ["0.00", "0.50", "+1.00"], "r": 2, "exp": "Una relación proporcional directa perfecta equivale a r = +1.00."},
            {"p": "¿Qué sucede con el valor de r si se cambian las unidades de medida de X o Y (ej. de metros a centímetros)?", "o": ["El valor de r no cambia (es adimensional)", "El valor de r se multiplica por 100", "El valor de r se vuelve negativo"], "r": 0, "exp": "El coeficiente r es una métrica adimensional insensible al cambio de escala lineal."}
        ],
        "4.3 Coeficiente de Determinación (R²)": [
            {"p": "Ejercicio Práctico: Si el coeficiente de correlación de Pearson es r = 0.80, ¿cuál es el valor del Coeficiente de Determinación (R²)?", "o": ["64.0%", "80.0%", "40.0%"], "r": 0, "exp": "R² = (0.80)² = 0.64 = 64.0%."},
            {"p": "Ejercicio Práctico: Si r = -0.90, ¿cuál es el porcentaje de variabilidad explicada (R²)?", "o": ["-81.0%", "81.0%", "90.0%"], "r": 1, "exp": "R² = (-0.90)² = +0.81 = 81.0% (R² siempre es positivo)."},
            {"p": "¿Cómo se interpreta un R² = 0.85 (85%) en un modelo de física?", "o": ["El 85% de la variación de Y es explicada por la variable X a través de la recta", "El error del modelo es del 85%", "Existe un 15% de probabilidad de que el cálculo sea correcto"], "r": 0, "exp": "R² cuantifica la proporción de la varianza total de Y que ajusta el modelo lineal."},
            {"p": "Ejercicio Práctico: Si R² = 0.49, ¿cuál es el módulo o valor absoluto de r?", "o": ["0.70", "0.49", "0.24"], "r": 0, "exp": "|r| = √0.49 = 0.70."},
            {"p": "¿Cuál es el rango de valores posibles para el Coeficiente de Determinación R²?", "o": ["De -1.0 a +1.0", "De 0.0 a 1.0 (ó 0% a 100%)", "Sin límite superior"], "r": 1, "exp": "Al ser un cuadrado de r, R² abarca de 0.0 (0%) a 1.0 (100%)."}
        ]
    }

    # ---------------------------------------------------------
    # GENERADOR DE EJEMPLOS RESUELTOS DINÁMICOS
    # ---------------------------------------------------------
    def generar_ejemplo_dinamico(subtema):
        if "1.1" in subtema:
            datos = sorted([random.randint(10, 50) for _ in range(6)])
            media = np.mean(datos)
            mediana = np.median(datos)
            df = pd.DataFrame({"Posición": [f"Dato {j+1}" for j in range(len(datos))], "Valor": datos})
            
            fig = px.histogram(x=datos, nbins=6, title="Distribución de la Muestra", labels={'x': 'Valores', 'y': 'Frecuencia'}, color_discrete_sequence=['#183059'])
            fig.update_layout(template="plotly_white")
            
            return df, [rf"\bar{{x}} = \frac{{{sum(datos)}}}{{{len(datos)}}} = \mathbf{{{media:.2f}}}", rf"\text{{Mediana}} = \mathbf{{{mediana:.2f}}}"], fig

        elif "1.2" in subtema:
            datos = sorted([random.randint(5, 30) for _ in range(5)])
            var = np.var(datos, ddof=1)
            std = np.std(datos, ddof=1)
            df = pd.DataFrame({"Muestra": [f"Punto {j+1}" for j in range(len(datos))], "Medición": datos})
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, 6)), y=datos, mode='markers+lines', name='Datos', marker=dict(size=10, color='#ef4444')))
            fig.add_hline(y=np.mean(datos), line_dash="dash", line_color="#183059", annotation_text=f"Media: {np.mean(datos):.2f}")
            fig.update_layout(title="Dispersión respecto a la Media", template="plotly_white", xaxis_title="Muestra", yaxis_title="Valores")
            
            return df, [rf"s^2 = \mathbf{{{var:.4f}}}", rf"s = \sqrt{{s^2}} = \mathbf{{{std:.4f}}}"], fig

        elif "1.3" in subtema:
            v_real = 100.0
            v_med = round(100.0 + random.uniform(-4.0, 4.0), 2)
            err_abs = abs(v_med - v_real)
            err_rel = (err_abs / v_real) * 100
            df = pd.DataFrame({"Parámetro": ["Valor Real", "Valor Medido"], "Medición": [v_real, v_med]})
            
            fig = px.bar(df, x="Parámetro", y="Medición", title="Comparativa de Error de Medición", color="Parámetro", color_discrete_sequence=['#183059', '#ef4444'])
            fig.update_layout(template="plotly_white")
            
            return df, [rf"\text{{Error Absoluto}} = |{v_med} - {v_real}| = \mathbf{{{err_abs:.2f}}}", rf"\text{{Error Relativo \%}} = \frac{{{err_abs:.2f}}}{{{v_real}}} \times 100 = \mathbf{{{err_rel:.2f}\%}}"], fig

        elif "2." in subtema:
            datos = sorted([random.randint(10, 80) for _ in range(15)])
            n = len(datos)
            k = int(1 + 3.322 * np.log10(n))
            counts, bins = np.histogram(datos, bins=k)
            df_st = pd.DataFrame({"Clase Min": bins[:-1].round(1), "Clase Max": bins[1:].round(1), "Frecuencia (f)": counts, "Frec. Acumulada (F)": np.cumsum(counts)})
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_st["Clase Max"], y=df_st["Frecuencia (f)"], name="Histograma", marker_color='#183059'))
            fig.add_trace(go.Scatter(x=df_st["Clase Max"], y=df_st["Frec. Acumulada (F)"], name="Ojiva", mode='lines+markers', line=dict(color='#ef4444', width=3)))
            fig.update_layout(title="Histograma de Frecuencia y Ojiva", template="plotly_white", xaxis_title="Intervalos", yaxis_title="Frecuencia")
            
            return df_st, [rf"K = 1 + 3.322 \log_{{10}}({n}) \implies K = \mathbf{{{k}\ \text{{clases}}}}"], fig

        elif "3." in subtema:
            mu, sig = 70, 10
            x_v = random.randint(55, 85)
            z_v = (x_v - mu) / sig
            prob = stats.norm.cdf(z_v)
            df = pd.DataFrame({"Variable": ["Media (μ)", "Desviación (σ)", "Evaluado (X)"], "Valor": [mu, sig, x_v]})
            
            x_ax = np.linspace(30, 110, 150)
            y_ax = stats.norm.pdf(x_ax, mu, sig)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_ax, y=y_ax, mode='lines', name='Normal', line=dict(color='#183059', width=3)))
            x_sh = np.linspace(30, x_v, 100)
            fig.add_trace(go.Scatter(x=np.concatenate([[x_sh[0]], x_sh, [x_sh[-1]]]), y=np.concatenate([[0], stats.norm.pdf(x_sh, mu, sig), [0]]), fill='tozeroy', fillcolor='rgba(43, 82, 152, 0.4)', name='Área Probabilidad'))
            fig.update_layout(title="Distribución Normal y Área Evaluada", template="plotly_white", xaxis_title="X", yaxis_title="Densidad")
            
            return df, [rf"Z = \frac{{{x_v} - {mu}}}{{{sig}}} = \mathbf{{{z_v:.2f}}}", rf"P(X \le {x_v}) = \mathbf{{{prob*100:.2f}\%}}"], fig

        else:
            x_v = [1, 2, 3, 4, 5]
            y_v = [round(2.0 * x + random.uniform(0.5, 2.0), 1) for x in x_v]
            slope, inter, r_v, p_v, std_e = stats.linregress(x_v, y_v)
            df = pd.DataFrame({"X": x_v, "Y": y_v})
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_v, y=y_v, mode='markers', name='Datos', marker=dict(size=10, color='#ef4444')))
            fig.add_trace(go.Scatter(x=np.array([1, 5]), y=inter + slope * np.array([1, 5]), mode='lines', name='Ajuste', line=dict(color='#183059', width=3)))
            fig.update_layout(title="Ajuste de Regresión Lineal", template="plotly_white", xaxis_title="X", yaxis_title="Y")
            
            return df, [rf"\hat{{Y}} = \mathbf{{{inter:.2f}}} + \mathbf{{{slope:.2f}}}X", rf"r = \mathbf{{{r_v:.4f}}} \quad | \quad R^2 = \mathbf{{{(r_v**2):.4f}}}"], fig

    # ---------------------------------------------------------
    # SELECCIÓN DEL SUBTEMA Y PANEL PRINCIPAL
    # ---------------------------------------------------------
    opciones_temas = {
        "📊 Módulo 1: Estadística Descriptiva y Mediciones": [
            "1.1 Medidas de Tendencia Central (Media, Mediana, Moda)",
            "1.2 Medidas de Dispersión (Varianza, Desviación Estándar)",
            "1.3 Error Experimental y Manejo de Unidades en Física"
        ],
        "📈 Módulo 2: Tablas de Frecuencia y Gráficas": [
            "2.1 Regla de Sturges y Construcción de Intervalos",
            "2.2 Frecuencia Absoluta, Relativa y Acumulada",
            "2.3 Histograma de Frecuencias y Ojiva Ascendente"
        ],
        "🎲 Módulo 3: Probabilidad y Distribución Normal": [
            "3.1 Propiedades de la Campana de Gauss (N(μ, σ))",
            "3.2 Puntuación Z y Estandarización de Variables",
            "3.3 Cálculo de Probabilidades Acumuladas P(X ≤ x)"
        ],
        "📉 Módulo 4: Regresión Lineal y Correlación": [
            "4.1 Diagrama de Dispersión y Recta de Ajuste",
            "4.2 Coeficiente de Correlación de Pearson (r)",
            "4.3 Coeficiente de Determinación (R²)"
        ]
    }

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        tema_principal = st.selectbox("📚 Selecciona el Tema Principal:", list(opciones_temas.keys()))
    with col_s2:
        subtema_seleccionado = st.selectbox("🎯 Selecciona el Subtema Específico:", opciones_temas[tema_principal])

    st.markdown("---")
    
    if st.button("📖 Generar lección acerca del tema seleccionado"):
        info_t = TEORIA_SUBTEMAS.get(subtema_seleccionado, {})
        
        st.markdown("### 1. Breve Historia")
        st.write(info_t.get("historia", ""))
        
        st.markdown("### 2. Teoría Fundamental")
        st.write(info_t.get("teoria", ""))
        
        st.markdown("### 3. Fórmulas Principales")
        for f in info_t.get("formulas", []):
            st.latex(f)
            
        st.markdown("### 4. Ejemplo Resuelto Paso a Paso")
        df_ex, sol_ex, fig_ex = generar_ejemplo_dinamico(subtema_seleccionado)
        
        st.write("**Tabla de Datos del Ejemplo:**")
        st.dataframe(df_ex.T, use_container_width=True)
        
        st.write("**Sustitución y Resultado:**")
        for s_eq in sol_ex:
            st.latex(s_eq)
            
        st.plotly_chart(fig_ex, use_container_width=True)

    if st.button("🔄 Generar otro ejemplo"):
        st.markdown("### 4. Nuevo Ejemplo Resuelto Generado")
        df_ex2, sol_ex2, fig_ex2 = generar_ejemplo_dinamico(subtema_seleccionado)
        st.dataframe(df_ex2.T, use_container_width=True)
        for s_eq in sol_ex2:
            st.latex(s_eq)
        st.plotly_chart(fig_ex2, use_container_width=True)

    # ---------------------------------------------------------
    # SECCIÓN QUIZ INTERACTIVO CON CONFIRMACIÓN PREVIA
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🎯 Quiz Interactivo de Autoevaluación")
    
    evaluar = st.radio(
        "¿Deseas realizar un quiz de autoevaluación (5 preguntas) sobre este tema?",
        ["No, prefiero repasar primero", "Sí, iniciar evaluación"],
        key=f"eval_choice_{subtema_seleccionado}"
    )

    if evaluar == "Sí, iniciar evaluación":
        preguntas_q = QUIZZES.get(subtema_seleccionado, [])
        if preguntas_q:
            st.info("📝 Responde las siguientes 5 preguntas (incluye teoría y ejercicios prácticos de cálculo):")
            respuestas_usuario = []
            for idx, q in enumerate(preguntas_q):
                st.markdown(f"**Pregunta {idx+1}:** {q['p']}")
                opcion_elegida = st.radio(
                    f"Selecciona tu respuesta para la pregunta {idx+1}:", 
                    q['o'], 
                    key=f"q_{subtema_seleccionado}_{idx}"
                )
                respuestas_usuario.append(q['o'].index(opcion_elegida))

            if st.button("🚀 Evaluar Respuestas y Obtener Puntaje"):
                correctas = 0
                total_p = len(preguntas_q)
                st.markdown("#### 📋 Retroalimentación Detallada")
                
                for idx, q in enumerate(preguntas_q):
                    if respuestas_usuario[idx] == q['r']:
                        correctas += 1
                        st.success(f"✅ **Pregunta {idx+1}: Correcto.** {q['exp']}")
                    else:
                        st.error(f"❌ **Pregunta {idx+1}: Incorrecto.** Respuesta correcta: **{q['o'][q['r']]}**. {q['exp']}")

                porcentaje = (correctas / total_p) * 100
                st.markdown("---")
                st.metric(label="📊 Porcentaje de Dominio del Tema", value=f"{porcentaje:.1f}%", delta=f"{correctas} de {total_p} correctas")
                
                if porcentaje == 100:
                    st.balloons()
                    st.success("🎉 ¡Excelente! Has demostrado un dominio completo y perfecto de este tema.")
                elif porcentaje >= 60:
                    st.info("👍 Buen desempeño. Revisa la retroalimentación para perfeccionar los detalles.")
                else:
                    st.warning("⚠️ Te recomendamos volver a repasar la lección teórica y hacer consultas en el chat inferior.")

    # ---------------------------------------------------------
    # SECCIÓN CHAT INTERACTIVO CON IA
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("💬 Chat Interactivo con IA")
    st.write("¿Tienes alguna duda sobre la lección? Pregúntale directamente al Asistente Virtual:")

    texto_usuario = st.chat_input(f"Escribe tu duda sobre {subtema_seleccionado}...")

    if texto_usuario:
        with st.chat_message("user"):
            st.write(texto_usuario)

        with st.chat_message("assistant"):
            try:
                st.write_stream(generar_stream_con_fallback(texto_usuario))
            except Exception as e:
                st.error(f"Error al procesar con la IA: {e}")
