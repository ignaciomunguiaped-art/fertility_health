import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics  # igual que la versión que funciona

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea", page_icon="🚕")
st.title("Aprendizaje en línea con River (Step-by-step desde GCS)")

st.markdown("""
Este panel permite entrenar un modelo de **aprendizaje incremental** con River,
procesando **un archivo por clic** desde Google Cloud Storage (GCS).
""")

# =========================================================
# RUTAS GCS
# =========================================================
MODEL_PATH   = "models/model_incremental.pkl"
HISTORY_PATH = "models/history_incremental.pkl"

# =========================================================
# PARÁMETROS
# =========================================================
bucket_name = st.text_input("Bucket de GCS:", "ml_big_data")
prefix      = st.text_input("Prefijo/carpeta:", "tlc_yellow_trips_2022/")
limite      = st.number_input("Filas a procesar por archivo:", value=1000, step=100)

st.markdown("---")

# =========================================================
# FUNCIONES GCS
# =========================================================
def save_model_to_gcs(model, bkt):
    try:
        storage.Client().bucket(bkt).blob(MODEL_PATH).upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en GCS: `{MODEL_PATH}`")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")

def load_model_from_gcs(bkt):
    try:
        blob = storage.Client().bucket(bkt).blob(MODEL_PATH)
        if blob.exists():
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(blob.download_as_bytes())
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo: {e}")
    return None

def save_history_to_gcs(bkt):
    data = {
        "history_r2":      st.session_state.history,
        "history_file":    st.session_state.history_file,
        "processed_files": st.session_state.processed_files,
        "index":           st.session_state.index,
    }
    try:
        storage.Client().bucket(bkt).blob(HISTORY_PATH).upload_from_string(pickle.dumps(data))
    except Exception as e:
        st.warning(f"No se pudo guardar historial: {e}")

def load_history_from_gcs(bkt):
    try:
        blob = storage.Client().bucket(bkt).blob(HISTORY_PATH)
        if blob.exists():
            return pickle.loads(blob.download_as_bytes())
    except Exception:
        pass
    return None

def delete_blob(bkt, path):
    try:
        blob = storage.Client().bucket(bkt).blob(path)
        if blob.exists():
            blob.delete()
    except Exception:
        pass

# =========================================================
# BOTÓN REINICIAR
# =========================================================
if st.button("🗑️ Reiniciar entrenamiento y borrar modelo guardado"):
    delete_blob(bucket_name, MODEL_PATH)
    delete_blob(bucket_name, HISTORY_PATH)
    st.session_state.clear()
    st.success("Entrenamiento reiniciado correctamente.")

# =========================================================
# INICIALIZAR SESSION STATE
# Detecta cambio de bucket igual que antes
# =========================================================
if (
    "loaded_bucket" not in st.session_state
    or st.session_state.loaded_bucket != bucket_name
):
    model = load_model_from_gcs(bucket_name)
    if model is None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()

    hist = load_history_from_gcs(bucket_name)

    st.session_state.model            = model
    st.session_state.metric           = metrics.R2()
    st.session_state.history          = hist["history_r2"]      if hist else []
    st.session_state.history_file     = hist["history_file"]    if hist else []
    st.session_state.processed_files  = hist["processed_files"] if hist else []
    st.session_state.index            = hist["index"]           if hist else 0
    st.session_state.blobs            = None
    st.session_state.loaded_bucket    = bucket_name

    if hist:
        st.info(f"Historial recuperado: {hist['index']} archivos procesados previamente.")

model  = st.session_state.model
metric = st.session_state.metric

# =========================================================
# FEATURE ENGINEERING — idéntico a la versión que funciona
# =========================================================
def _parse_time_fields(row):
    if "pickup_hour" in row and pd.notna(row["pickup_hour"]):
        try:
            hour = int(pd.to_numeric(row["pickup_hour"], errors="coerce"))
            return None, max(0, min(hour, 23))
        except Exception:
            pass
    for c in ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"):
        if c in row and pd.notna(row[c]):
            dt = pd.to_datetime(row[c], errors="coerce", utc=False)
            if pd.notna(dt):
                return dt, int(dt.hour)
    return None, 0

def _extract_x(row):
    dist = float(pd.to_numeric(row.get("trip_distance", 0), errors="coerce") or 0)
    psg  = float(pd.to_numeric(row.get("passenger_count", 0), errors="coerce") or 0)
    dt, hour = _parse_time_fields(row)
    dow     = int(dt.weekday()) if isinstance(dt, pd.Timestamp) else 0
    weekend = 1.0 if dow >= 5 else 0.0
    return {
        "dist":       dist,
        "log_dist":   float(np.log1p(max(dist, 0))),
        "pass":       psg,
        "hour":       float(hour),
        "dow":        float(dow),
        "is_weekend": weekend,
    }

def _valid_target(v):
    y = pd.to_numeric(v, errors="coerce")
    if pd.isna(y):
        return None
    return float(y)

# =========================================================
# PROCESAR — idéntico a la versión que funciona
# =========================================================
def process_single_blob(bkt, blob_name, limite=1000, chunksize=500):
    if blob_name.endswith("/") or not blob_name.endswith(".csv"):
        return None

    blob = storage.Client().bucket(bkt).blob(blob_name)

    try:
        content = blob.download_as_bytes()
        buffer  = io.BytesIO(content)
        count   = 0

        for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):
            if not {"trip_distance", "passenger_count", "fare_amount"}.issubset(chunk.columns):
                continue

            for col in ["trip_distance", "passenger_count", "fare_amount"]:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
            chunk = chunk[
                chunk["fare_amount"].between(2, 200) &
                chunk["trip_distance"].between(0.1, 50) &
                chunk["passenger_count"].between(1, 6)
            ]

            for _, row in chunk.iterrows():
                if count >= limite:
                    break
                y = _valid_target(row["fare_amount"])
                if y is None:
                    continue
                x    = _extract_x(row)
                pred = model.predict_one(x)
                model.learn_one(x, y)
                metric.update(y, pred)
                count += 1

    except Exception as e:
        st.warning(f"Error en {blob_name}: {e}")
        return None

    return metric.get()

# =========================================================
# BOTÓN: PROCESAR SIGUIENTE ARCHIVO
# =========================================================
st.subheader("Procesamiento incremental")

if st.button("▶️ Procesar siguiente archivo"):

    if st.session_state.blobs is None:
        client = storage.Client()
        blobs  = list(client.bucket(bucket_name).list_blobs(prefix=prefix))
        st.session_state.blobs = blobs
        st.info(f"Se encontraron {len(blobs)} archivos.")

    blobs = st.session_state.blobs
    idx   = st.session_state.index

    if idx >= len(blobs):
        st.success("✅ Todos los archivos ya fueron procesados.")
    else:
        blob  = blobs[idx]
        short = blob.name.split("/")[-1]

        if not short or not blob.name.endswith(".csv"):
            st.info(f"Saltando: `{blob.name}`")
        else:
            st.write(f"Procesando {idx+1}/{len(blobs)}: `{short}`")
            score = process_single_blob(bucket_name, blob.name, int(limite))

            if score is not None:
                st.session_state.history.append(score)
                st.session_state.history_file.append(short)
                st.session_state.processed_files.append(short)
                st.write(f"R² acumulado: **{score:.4f}**")

                save_model_to_gcs(model, bucket_name)
                save_history_to_gcs(bucket_name)

        st.session_state.index += 1

# =========================================================
# ESTADO ACTUAL
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")

last_r2 = st.session_state.history[-1] if st.session_state.history else 0.0
st.write(f"Archivos procesados: **{st.session_state.index}**")
st.write(f"R² acumulado actual: **{last_r2:.4f}**")

if st.session_state.history:
    df_hist = pd.DataFrame({
        "archivo":      st.session_state.processed_files,
        "R2_acumulado": st.session_state.history,
    })

    st.subheader("Historial de procesamiento")
    st.dataframe(df_hist)

    st.subheader("Evolución R² acumulado")
    st.line_chart(df_hist[["R2_acumulado"]])

st.caption("Cloud Run + River • Dataset público de taxis NYC")





