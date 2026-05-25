import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics, optim

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea", page_icon="🚕")
st.title("Aprendizaje en línea con River (Step-by-step desde GCS)")

st.markdown("""
Este panel permite entrenar un modelo de **aprendizaje incremental** con River,
procesando **un archivo por clic** desde Google Cloud Storage (GCS).

La lógica usa evaluación progresiva: primero se predice, luego se actualiza el modelo.
""")

# =========================================================
# RUTAS EN GCS
# =========================================================
MODEL_PATH   = "models/model_incremental.pkl"
HISTORY_PATH = "models/history_incremental.pkl"

# =========================================================
# FUNCIONES AUXILIARES GCS — MODELO
# =========================================================
def save_model_to_gcs(model, bucket_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(MODEL_PATH)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en GCS: `{MODEL_PATH}`")
    except Exception as e:
        st.error(f"Error al guardar modelo ({type(e).__name__}): {e}")


def load_model_from_gcs(bucket_name):
    try:
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(MODEL_PATH)
        if blob.exists():
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(blob.download_as_bytes())
        return None
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo previo: {e}")
        return None


def delete_model_from_gcs(bucket_name):
    try:
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(MODEL_PATH)
        if blob.exists():
            blob.delete()
            st.success("Modelo eliminado de GCS.")
        else:
            st.info("No había modelo guardado en GCS.")
    except Exception as e:
        st.warning(f"No se pudo eliminar el modelo: {e}")


# =========================================================
# FUNCIONES AUXILIARES GCS — HISTORIAL
# =========================================================
def save_history_to_gcs(bucket_name):
    data = {
        "processed_files":  st.session_state.processed_files,
        "history_r2":       st.session_state.history_r2,
        "history_mae":      st.session_state.history_mae,
        "history_file_r2":  st.session_state.history_file_r2,
        "history_file_mae": st.session_state.history_file_mae,
        "index":            st.session_state.index,
    }
    try:
        client = storage.Client()
        client.bucket(bucket_name).blob(HISTORY_PATH).upload_from_string(pickle.dumps(data))
    except Exception as e:
        st.warning(f"No se pudo guardar historial: {e}")


def load_history_from_gcs(bucket_name):
    try:
        blob = storage.Client().bucket(bucket_name).blob(HISTORY_PATH)
        if blob.exists():
            return pickle.loads(blob.download_as_bytes())
    except Exception as e:
        st.warning(f"No se pudo cargar historial: {e}")
    return None


def delete_history_from_gcs(bucket_name):
    try:
        blob = storage.Client().bucket(bucket_name).blob(HISTORY_PATH)
        if blob.exists():
            blob.delete()
    except Exception as e:
        st.warning(f"No se pudo eliminar historial: {e}")


# =========================================================
# MODELO
# =========================================================
def new_model():
    """
    Modelo incremental conservador.
    Learning rate pequeño para evitar que los pesos se disparen
    al inicio del entrenamiento.
    """
    return preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(0.001),
        intercept_lr=0.001
    )


# =========================================================
# PARÁMETROS
# =========================================================
bucket_name = st.text_input("Bucket de GCS:", "ml_big_data")
prefix      = st.text_input("Prefijo/carpeta:", "tlc_yellow_trips_2022/")
limite      = st.number_input("Filas a procesar por archivo:", value=1000, step=100)

st.markdown("---")

# =========================================================
# BOTÓN PARA REINICIAR TODO
# =========================================================
if st.button("🗑️ Reiniciar entrenamiento y borrar modelo guardado"):
    delete_model_from_gcs(bucket_name)
    delete_history_from_gcs(bucket_name)

    st.session_state.model            = new_model()
    st.session_state.metric_r2        = metrics.R2()
    st.session_state.metric_mae       = metrics.MAE()
    st.session_state.history_r2       = []
    st.session_state.history_mae      = []
    st.session_state.history_file_r2  = []
    st.session_state.history_file_mae = []
    st.session_state.processed_files  = []
    st.session_state.blobs            = None
    st.session_state.index            = 0

    st.success("Entrenamiento reiniciado correctamente.")

# =========================================================
# INICIALIZAR SESSION STATE
# Recupera modelo e historial desde GCS si Cloud Run reinició.
# =========================================================
if "model" not in st.session_state:

    # --- Modelo ---
    loaded_model = load_model_from_gcs(bucket_name)
    if loaded_model is None:
        loaded_model = new_model()
    st.session_state.model = loaded_model

    # --- Historial persistido ---
    hist = load_history_from_gcs(bucket_name)
    if hist:
        st.session_state.processed_files  = hist["processed_files"]
        st.session_state.history_r2       = hist["history_r2"]
        st.session_state.history_mae      = hist["history_mae"]
        st.session_state.history_file_r2  = hist["history_file_r2"]
        st.session_state.history_file_mae = hist["history_file_mae"]
        st.session_state.index            = hist["index"]
        st.info(f"Historial recuperado: {hist['index']} archivos procesados previamente.")
    else:
        st.session_state.processed_files  = []
        st.session_state.history_r2       = []
        st.session_state.history_mae      = []
        st.session_state.history_file_r2  = []
        st.session_state.history_file_mae = []
        st.session_state.index            = 0

    # Las métricas de River no se serializan de forma confiable;
    # se reconstruyen en memoria y se muestran desde el historial guardado.
    st.session_state.metric_r2  = metrics.R2()
    st.session_state.metric_mae = metrics.MAE()
    st.session_state.blobs      = None

model     = st.session_state.model
metric_r2 = st.session_state.metric_r2
metric_mae= st.session_state.metric_mae

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def _parse_time_fields(row):
    """Extrae fecha y hora a partir de distintas columnas posibles."""
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
    """Extrae las variables predictoras del viaje."""
    dist = float(row["trip_distance"])
    psg  = float(row["passenger_count"])

    dt, hour = _parse_time_fields(row)
    dow      = int(dt.weekday()) if isinstance(dt, pd.Timestamp) else 0
    weekend  = 1.0 if dow >= 5 else 0.0

    return {
        "dist":      dist,
        "log_dist":  float(np.log1p(max(dist, 0))),
        "pass":      psg,
        "hour":      float(hour),
        "dow":       float(dow),
        "is_weekend": weekend,
    }


# =========================================================
# PROCESAR UN SOLO ARCHIVO
# =========================================================
def process_single_blob(bucket_name, blob_name, limite=1000, chunksize=500):

    client = storage.Client()
    blob   = client.bucket(bucket_name).blob(blob_name)

    chunks_validos = []

    try:
        content = blob.download_as_bytes()
        buffer  = io.BytesIO(content)

        for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):

            cols_needed = ["trip_distance", "passenger_count", "fare_amount"]

            if not set(cols_needed).issubset(chunk.columns):
                continue

            for col in cols_needed:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = chunk.dropna(subset=cols_needed)
            chunk = chunk[
                chunk["fare_amount"].between(2, 200) &
                chunk["trip_distance"].between(0.1, 50) &
                chunk["passenger_count"].between(1, 6)
            ]

            if not chunk.empty:
                chunks_validos.append(chunk)

        if not chunks_validos:
            return None

        df_file = pd.concat(chunks_validos, ignore_index=True)

        if len(df_file) > limite:
            df_file = df_file.sample(n=limite, random_state=42)

        file_r2  = metrics.R2()
        file_mae = metrics.MAE()
        count    = 0

        for _, row in df_file.iterrows():
            y    = float(row["fare_amount"])
            x    = _extract_x(row)
            pred = model.predict_one(x)

            pred_eval = float(np.clip(pred, 2, 200)) if pred is not None else 0.0

            metric_r2.update(y, pred_eval)
            metric_mae.update(y, pred_eval)
            file_r2.update(y, pred_eval)
            file_mae.update(y, pred_eval)

            model.learn_one(x, y)
            count += 1

    except Exception as e:
        st.error(f"Error en `{blob_name}` ({type(e).__name__}): {e}")
        return None

    return {
        "count":      count,
        "file_r2":    file_r2.get(),
        "file_mae":   file_mae.get(),
        "global_r2":  metric_r2.get(),
        "global_mae": metric_mae.get(),
    }


# =========================================================
# BOTÓN: PROCESAR SIGUIENTE ARCHIVO
# =========================================================
st.markdown("---")
st.subheader("Procesamiento incremental")

if st.button("▶️ Procesar siguiente archivo"):

    if st.session_state.blobs is None:
        client = storage.Client()
        blobs  = list(client.bucket(bucket_name).list_blobs(prefix=prefix))
        blobs  = [b for b in blobs if b.name.endswith(".csv") and not b.name.endswith("/")]

        st.session_state.blobs = blobs
        # No resetear index aquí; puede haber progreso previo cargado de GCS
        st.info(f"Se encontraron {len(blobs)} archivos CSV en `{prefix}`.")

    blobs = st.session_state.blobs
    idx   = st.session_state.index

    if idx >= len(blobs):
        st.success("✅ Todos los archivos ya fueron procesados.")

    else:
        blob  = blobs[idx]
        short = blob.name.split("/")[-1]

        st.write(f"Procesando archivo {idx + 1}/{len(blobs)}: `{short}`")

        result = process_single_blob(
            bucket_name=bucket_name,
            blob_name=blob.name,
            limite=int(limite),
        )

        if result is not None:
            st.session_state.history_r2.append(result["global_r2"])
            st.session_state.history_mae.append(result["global_mae"])
            st.session_state.history_file_r2.append(result["file_r2"])
            st.session_state.history_file_mae.append(result["file_mae"])
            st.session_state.processed_files.append(short)

            st.write(f"Registros procesados: **{result['count']}**")
            st.write(f"R² del archivo actual: **{result['file_r2']:.4f}**")
            st.write(f"MAE del archivo actual: **{result['file_mae']:.4f}**")
            st.write(f"R² acumulado: **{result['global_r2']:.4f}**")
            st.write(f"MAE acumulado: **{result['global_mae']:.4f}**")

            # Guardar modelo E historial en GCS
            save_model_to_gcs(model, bucket_name)
            save_history_to_gcs(bucket_name)

        else:
            st.warning("No se procesaron registros válidos en este archivo.")

        st.session_state.index += 1

# =========================================================
# ESTADO ACTUAL
# Se muestra desde el historial guardado, no desde metric_r2/mae
# que se reinician con cada instancia de Cloud Run.
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")

last_r2  = st.session_state.history_r2[-1]  if st.session_state.history_r2  else 0.0
last_mae = st.session_state.history_mae[-1] if st.session_state.history_mae else 0.0

st.write(f"Archivos procesados: **{st.session_state.index}**")
st.write(f"R² acumulado actual: **{last_r2:.4f}**")
st.write(f"MAE acumulado actual: **{last_mae:.4f}**")

# =========================================================
# HISTORIAL
# =========================================================
if st.session_state.history_r2:

    df_hist = pd.DataFrame({
        "archivo":      st.session_state.processed_files,
        "R2_archivo":   st.session_state.history_file_r2,
        "MAE_archivo":  st.session_state.history_file_mae,
        "R2_acumulado": st.session_state.history_r2,
        "MAE_acumulado":st.session_state.history_mae,
    })

    st.subheader("Historial de procesamiento")
    st.dataframe(df_hist)

    st.subheader("Evolución R² acumulado")
    st.line_chart(df_hist[["R2_acumulado"]])

    st.subheader("Evolución MAE acumulado")
    st.line_chart(df_hist[["MAE_acumulado"]])

    st.subheader("Métricas por archivo")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(df_hist[["R2_archivo"]])
    with col2:
        st.line_chart(df_hist[["MAE_archivo"]])

st.caption("Cloud Run + River • Dataset público de taxis NYC")




