import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import traceback
from google.cloud import storage
from river import linear_model, preprocessing, metrics, optim

st.set_page_config(page_title="Aprendizaje en línea", page_icon="🚕")
st.title("Aprendizaje en línea con River (Step-by-step desde GCS)")

MODEL_PATH   = "models/model_incremental.pkl"
HISTORY_PATH = "models/history_incremental.pkl"

bucket_name = st.text_input("Bucket de GCS:", "ml_big_data")
prefix      = st.text_input("Prefijo/carpeta:", "tlc_yellow_trips_2022/")
limite      = st.number_input("Filas a procesar por archivo:", value=1000, step=100)

st.markdown("---")

# =========================================================
# DIAGNÓSTICO — prueba GCS y primer CSV
# =========================================================
st.subheader("🔧 Diagnóstico")

col1, col2 = st.columns(2)

with col1:
    if st.button("1️⃣ Probar escritura en GCS"):
        log = []
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob   = bucket.blob("test_write.txt")
            blob.upload_from_string("ok")
            log.append("✅ Escritura en GCS: OK")
            blob.delete()
            log.append("✅ Lectura/borrado en GCS: OK")
        except Exception as e:
            log.append(f"❌ Error GCS: {type(e).__name__}: {e}")
        st.session_state.diag_gcs = log

with col2:
    if st.button("2️⃣ Inspeccionar primer CSV"):
        log = []
        try:
            client = storage.Client()
            blobs  = list(client.bucket(bucket_name).list_blobs(prefix=prefix))
            blobs  = [b for b in blobs if b.name.endswith(".csv")]
            if not blobs:
                log.append("❌ No se encontraron archivos .csv")
            else:
                b = blobs[0]
                log.append(f"📂 Archivo: {b.name}")
                log.append(f"📦 Tamaño: {b.size / 1024 / 1024:.1f} MB")

                # Descargar solo primeros 200KB para inspeccionar
                content = b.download_as_bytes(start=0, end=200_000)
                log.append(f"✅ Primeros 200KB descargados")

                df_sample = pd.read_csv(io.BytesIO(content), nrows=5, low_memory=False)
                log.append(f"📋 Columnas: {list(df_sample.columns)}")
                log.append(f"📋 Primeras filas:")
                for col in ["trip_distance", "passenger_count", "fare_amount"]:
                    if col in df_sample.columns:
                        log.append(f"  ✅ '{col}': {df_sample[col].tolist()}")
                    else:
                        log.append(f"  ❌ '{col}' NO ENCONTRADA")
        except Exception as e:
            log.append(f"❌ Error: {type(e).__name__}: {e}")
            log.append(traceback.format_exc())
        st.session_state.diag_csv = log

if st.session_state.get("diag_gcs"):
    st.markdown("**Resultado prueba GCS:**")
    for line in st.session_state.diag_gcs:
        st.write(line)

if st.session_state.get("diag_csv"):
    st.markdown("**Resultado inspección CSV:**")
    for line in st.session_state.diag_csv:
        st.write(line)

st.markdown("---")

# =========================================================
# FUNCIONES GCS
# =========================================================
def save_model_to_gcs(model, bkt):
    try:
        storage.Client().bucket(bkt).blob(MODEL_PATH).upload_from_string(pickle.dumps(model))
        return True, "✅ Modelo guardado en GCS"
    except Exception as e:
        return False, f"❌ Error al guardar modelo: {type(e).__name__}: {e}"

def load_model_from_gcs(bkt):
    try:
        blob = storage.Client().bucket(bkt).blob(MODEL_PATH)
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

def save_history_to_gcs(bkt):
    data = {
        "processed_files":  st.session_state.processed_files,
        "history_r2":       st.session_state.history_r2,
        "history_mae":      st.session_state.history_mae,
        "history_file_r2":  st.session_state.history_file_r2,
        "history_file_mae": st.session_state.history_file_mae,
        "index":            st.session_state.index,
    }
    try:
        storage.Client().bucket(bkt).blob(HISTORY_PATH).upload_from_string(pickle.dumps(data))
        return True
    except Exception:
        return False

def load_history_from_gcs(bkt):
    try:
        blob = storage.Client().bucket(bkt).blob(HISTORY_PATH)
        if blob.exists():
            return pickle.loads(blob.download_as_bytes())
    except Exception:
        pass
    return None

# =========================================================
# MODELO
# =========================================================
def new_model():
    return preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(0.001),
        intercept_lr=0.001
    )

def reset_session_state(bkt=None, load_from_gcs=False):
    model = None
    hist  = None
    if load_from_gcs and bkt:
        model = load_model_from_gcs(bkt)
        hist  = load_history_from_gcs(bkt)

    st.session_state.model            = model if model else new_model()
    st.session_state.metric_r2        = metrics.R2()
    st.session_state.metric_mae       = metrics.MAE()
    st.session_state.processed_files  = hist["processed_files"]  if hist else []
    st.session_state.history_r2       = hist["history_r2"]       if hist else []
    st.session_state.history_mae      = hist["history_mae"]      if hist else []
    st.session_state.history_file_r2  = hist["history_file_r2"]  if hist else []
    st.session_state.history_file_mae = hist["history_file_mae"] if hist else []
    st.session_state.index            = hist["index"]            if hist else 0
    st.session_state.blobs            = None
    st.session_state.loaded_bucket    = bkt
    st.session_state.last_log         = []

if st.button("🗑️ Reiniciar entrenamiento y borrar modelo guardado"):
    delete_blob(bucket_name, MODEL_PATH)
    delete_blob(bucket_name, HISTORY_PATH)
    reset_session_state(bkt=bucket_name, load_from_gcs=False)
    st.success("Entrenamiento reiniciado correctamente.")

if (
    "loaded_bucket" not in st.session_state
    or st.session_state.loaded_bucket != bucket_name
):
    reset_session_state(bkt=bucket_name, load_from_gcs=True)

model      = st.session_state.model
metric_r2  = st.session_state.metric_r2
metric_mae = st.session_state.metric_mae

# =========================================================
# FEATURE ENGINEERING
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
    dist     = float(row["trip_distance"])
    psg      = float(row["passenger_count"])
    dt, hour = _parse_time_fields(row)
    dow      = int(dt.weekday()) if isinstance(dt, pd.Timestamp) else 0
    return {
        "dist":       dist,
        "log_dist":   float(np.log1p(max(dist, 0))),
        "pass":       psg,
        "hour":       float(hour),
        "dow":        float(dow),
        "is_weekend": 1.0 if dow >= 5 else 0.0,
    }

# =========================================================
# PROCESAR
# =========================================================
def process_single_blob(bkt, blob_name, limite=1000, chunksize=500):
    log    = []
    chunks = []

    try:
        blob    = storage.Client().bucket(bkt).blob(blob_name)
        log.append(f"📦 Tamaño: {blob.size / 1024 / 1024:.1f} MB")

        content = blob.download_as_bytes()
        log.append(f"✅ Descargado: {len(content)/1024:.1f} KB")

        buffer      = io.BytesIO(content)
        cols_needed = ["trip_distance", "passenger_count", "fare_amount"]
        total_leidas = total_pre = total_post = 0

        for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):
            total_leidas += len(chunk)

            if not set(cols_needed).issubset(chunk.columns):
                if total_leidas == len(chunk):
                    log.append(f"⚠️ Columnas del CSV: {list(chunk.columns)}")
                continue

            for col in cols_needed:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_needed)
            total_pre += len(chunk)

            chunk = chunk[
                chunk["fare_amount"].between(2, 200) &
                chunk["trip_distance"].between(0.1, 50) &
                chunk["passenger_count"].between(1, 6)
            ]
            total_post += len(chunk)

            if not chunk.empty:
                chunks.append(chunk)

        log.append(f"📊 Filas leídas: {total_leidas}")
        log.append(f"📊 Filas tras NaN: {total_pre}")
        log.append(f"📊 Filas tras filtros: {total_post}")

        if not chunks:
            log.append("❌ Sin filas válidas.")
            return None, log

        df = pd.concat(chunks, ignore_index=True)
        if len(df) > limite:
            df = df.sample(n=limite, random_state=42)

        file_r2  = metrics.R2()
        file_mae = metrics.MAE()
        count    = 0

        for _, row in df.iterrows():
            y         = float(row["fare_amount"])
            x         = _extract_x(row)
            pred      = model.predict_one(x)
            pred_eval = float(np.clip(pred, 2, 200)) if pred is not None else 0.0
            metric_r2.update(y, pred_eval)
            metric_mae.update(y, pred_eval)
            file_r2.update(y, pred_eval)
            file_mae.update(y, pred_eval)
            model.learn_one(x, y)
            count += 1

        log.append(f"✅ Registros entrenados: {count}")
        log.append(f"R² archivo:    {file_r2.get():.4f}")
        log.append(f"MAE archivo:   {file_mae.get():.4f}")
        log.append(f"R² acumulado:  {metric_r2.get():.4f}")
        log.append(f"MAE acumulado: {metric_mae.get():.4f}")

        return {
            "count":      count,
            "file_r2":    file_r2.get(),
            "file_mae":   file_mae.get(),
            "global_r2":  metric_r2.get(),
            "global_mae": metric_mae.get(),
        }, log

    except Exception as e:
        log.append(f"❌ Excepción: {type(e).__name__}: {e}")
        log.append(traceback.format_exc())
        return None, log

# =========================================================
# BOTÓN PROCESAR
# =========================================================
st.subheader("Procesamiento incremental")

if st.button("▶️ Procesar siguiente archivo"):
    if st.session_state.blobs is None:
        try:
            blobs = list(storage.Client().bucket(bucket_name).list_blobs(prefix=prefix))
            blobs = [b for b in blobs if b.name.endswith(".csv") and not b.name.endswith("/")]
            st.session_state.blobs    = blobs
            st.session_state.last_log = [f"ℹ️ Encontrados {len(blobs)} archivos CSV."]
        except Exception as e:
            st.session_state.last_log = [f"❌ Error listando archivos: {type(e).__name__}: {e}"]

    if st.session_state.blobs is not None:
        blobs = st.session_state.blobs
        idx   = st.session_state.index

        if idx >= len(blobs):
            st.session_state.last_log = ["✅ Todos los archivos procesados."]
        else:
            blob  = blobs[idx]
            short = blob.name.split("/")[-1]
            st.session_state.last_log = [f"📂 Procesando {idx+1}/{len(blobs)}: `{short}`"]

            result, log = process_single_blob(bkt=bucket_name, blob_name=blob.name, limite=int(limite))
            st.session_state.last_log.extend(log)

            if result is not None:
                st.session_state.history_r2.append(result["global_r2"])
                st.session_state.history_mae.append(result["global_mae"])
                st.session_state.history_file_r2.append(result["file_r2"])
                st.session_state.history_file_mae.append(result["file_mae"])
                st.session_state.processed_files.append(short)

                ok, msg = save_model_to_gcs(model, bucket_name)
                st.session_state.last_log.append(msg)

                if save_history_to_gcs(bucket_name):
                    st.session_state.last_log.append("✅ Historial guardado en GCS.")
                else:
                    st.session_state.last_log.append("❌ Error guardando historial.")

            st.session_state.index += 1

# Log persistente
if st.session_state.get("last_log"):
    st.markdown("**Último resultado:**")
    for line in st.session_state.last_log:
        st.write(line)

# =========================================================
# ESTADO ACTUAL
# =========================================================
st.markdown("---")
st.subheader("Estado actual del modelo")

last_r2  = st.session_state.history_r2[-1]  if st.session_state.history_r2  else 0.0
last_mae = st.session_state.history_mae[-1] if st.session_state.history_mae else 0.0

st.write(f"Archivos procesados: **{st.session_state.index}**")
st.write(f"R² acumulado actual: **{last_r2:.4f}**")
st.write(f"MAE acumulado actual: **{last_mae:.4f}**")

if st.session_state.history_r2:
    df_hist = pd.DataFrame({
        "archivo":       st.session_state.processed_files,
        "R2_archivo":    st.session_state.history_file_r2,
        "MAE_archivo":   st.session_state.history_file_mae,
        "R2_acumulado":  st.session_state.history_r2,
        "MAE_acumulado": st.session_state.history_mae,
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





