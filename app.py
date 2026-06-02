
import os
import pickle
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
import logging
from google.cloud import storage
import tempfile

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.json.sort_keys = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cloud Storage Configuration
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'fertility_health')
PROJECT_ID = os.environ.get('GCP_PROJECT', 'ml-big-data-ignacio')

# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES PARA CLOUD STORAGE
# ════════════════════════════════════════════════════════════════════════════

def descargar_de_gcs(bucket_name, file_path, local_path):
    """
    Descargar archivo de Cloud Storage
    
    Args:
        bucket_name: Nombre del bucket
        file_path: Ruta del archivo en GCS
        local_path: Ruta local donde guardar
    """
    try:
        logger.info(f"📥 Descargando {file_path} desde GCS...")
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        blob.download_to_filename(local_path)
        logger.info(f"✅ Descargado: {local_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error descargando {file_path}: {str(e)}")
        return False

def cargar_modelos_desde_gcs():
    """
    Cargar modelos desde Cloud Storage
    Retorna: (xgb_model, encoders, le_target)
    """
    xgb_model = None
    encoders = None
    le_target = None
    
    # Crear directorio temporal para modelos
    temp_dir = tempfile.gettempdir()
    
    try:
        # Descargar modelo XGBoost
        xgb_local = os.path.join(temp_dir, 'modelo_xgboost_optuna.pkl')
        if descargar_de_gcs(GCS_BUCKET, 'modelos/modelo_xgboost_optuna.pkl', xgb_local):
            with open(xgb_local, 'rb') as f:
                xgb_model = pickle.load(f)
            logger.info("✅ Modelo XGBoost cargado desde GCS")
        
        # Descargar encoders
        encoders_local = os.path.join(temp_dir, 'label_encoders.pkl')
        if descargar_de_gcs(GCS_BUCKET, 'modelos/label_encoders.pkl', encoders_local):
            with open(encoders_local, 'rb') as f:
                encoders = pickle.load(f)
            logger.info("✅ Encoders cargados desde GCS")
        
        # Descargar encoder del target
        target_local = os.path.join(temp_dir, 'label_encoder_target.pkl')
        if descargar_de_gcs(GCS_BUCKET, 'modelos/label_encoder_target.pkl', target_local):
            with open(target_local, 'rb') as f:
                le_target = pickle.load(f)
            logger.info("✅ Target encoder cargado desde GCS")
        
        return xgb_model, encoders, le_target
    
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {str(e)}")
        return None, None, None

# ════════════════════════════════════════════════════════════════════════════
# MODELOS (lazy loading — se cargan en la primera predicción)
# ════════════════════════════════════════════════════════════════════════════

xgb_model = None
encoders = None
le_target = None
_models_loaded = False

def _ensure_models():
    global xgb_model, encoders, le_target, _models_loaded
    if _models_loaded:
        return
    logger.info(f"🔄 Cargando modelos desde GCS (bucket: {GCS_BUCKET})...")
    xgb_model, encoders, le_target = cargar_modelos_desde_gcs()
    _models_loaded = True
    if xgb_model is None:
        logger.warning("⚠️ No se pudo cargar el modelo.")

# ════════════════════════════════════════════════════════════════════════════
# VARIABLES Y CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'Female_Age', 'Male_Age', 'BMI', 'Menstrual_Regularity', 'PCOS',
    'Stress_Level', 'Smoking', 'Alcohol_Intake', 'Sperm_Count_Million_per_ml',
    'Motility_%', 'Trying_Duration_Months', 'Treatment_Type'
]

CATEGORICAL_OPTIONS = {
    'Menstrual_Regularity': ['Regular', 'Irregular'],
    'PCOS': ['No', 'Yes'],
    'Stress_Level': ['Low', 'Medium', 'High'],
    'Smoking': [0, 1],
    'Alcohol_Intake': ['Moderate', 'High', 'None'],
    'Treatment_Type': ['IVF', 'Medication', 'None']
}

RANGES = {
    'Female_Age': (20, 50),
    'Male_Age': (20, 70),
    'BMI': (15, 40),
    'Sperm_Count_Million_per_ml': (0, 200),
    'Motility_%': (0, 100),
    'Trying_Duration_Months': (1, 120)
}

# ════════════════════════════════════════════════════════════════════════════
# RUTAS - FRONTEND
# ════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Página principal - Formulario de entrada"""
    return render_template('index.html', 
                         categorical_options=CATEGORICAL_OPTIONS)

@app.route('/info')
def info():
    """Página de información sobre el modelo"""
    return render_template('info.html')

# ════════════════════════════════════════════════════════════════════════════
# RUTAS - API
# ════════════════════════════════════════════════════════════════════════════

@app.route('/api/prediccion', methods=['POST'])
def hacer_prediccion():
    """
    Endpoint para hacer predicciones
    Recibe: JSON con datos del usuario
    Retorna: JSON con predicción y probabilidades
    """
    try:
        _ensure_models()
        # Validar que el modelo esté cargado
        if xgb_model is None:
            return jsonify({
                'error': 'Modelo no disponible. Verifica la conexión con Cloud Storage.',
                'status': 503
            }), 503
        
        # Obtener datos del request
        data = request.get_json()
        logger.info(f"📥 Datos recibidos: {list(data.keys())}")
        
        # Validar que tenemos todos los campos
        campos_faltantes = [f for f in FEATURE_NAMES if f not in data]
        if campos_faltantes:
            return jsonify({
                'error': f'Campos faltantes: {", ".join(campos_faltantes)}',
                'status': 400
            }), 400
        
        # Crear DataFrame
        df_input = pd.DataFrame([data])
        
        # Validar rangos
        validacion = validar_datos(df_input)
        if not validacion['valido']:
            return jsonify({
                'error': validacion['mensaje'],
                'status': 400
            }), 400
        
        # Encoding de variables categóricas
        df_encoded = df_input.copy()
        if encoders:
            for col in df_encoded.select_dtypes(include='object').columns:
                if col in encoders:
                    try:
                        df_encoded[col] = encoders[col].transform(df_encoded[col])
                    except ValueError as e:
                        return jsonify({
                            'error': f'Valor inválido en {col}: {str(e)}',
                            'status': 400
                        }), 400
        
        # Predicción
        prediccion = xgb_model.predict(df_encoded)[0]
        probabilidades = xgb_model.predict_proba(df_encoded)[0]
        
        # Decodificar predicción
        if le_target is not None:
            prediccion_label = le_target.classes_[prediccion]
        else:
            prediccion_label = 'Success' if prediccion == 1 else 'Failure'
        
        # Retornar resultado
        resultado = {
            'prediccion': prediccion_label,
            'probabilidad_failure': float(probabilidades[0]),
            'probabilidad_success': float(probabilidades[1]),
            'confianza': float(max(probabilidades)),
            'status': 200,
            'mensaje': 'Predicción realizada exitosamente'
        }
        
        logger.info(f"✅ Predicción: {resultado['prediccion']} (confianza: {resultado['confianza']:.2%})")
        return jsonify(resultado), 200
    
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 500
        }), 500

@app.route('/api/variables', methods=['GET'])
def obtener_variables():
    """Retorna información sobre las variables para el frontend"""
    return jsonify({
        'features': FEATURE_NAMES,
        'categorical_options': CATEGORICAL_OPTIONS,
        'ranges': RANGES,
        'status': 200
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check para Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': xgb_model is not None,
        'encoders_loaded': encoders is not None,
        'target_encoder_loaded': le_target is not None
    }), 200

# ════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════

def validar_datos(df):
    """Validar rangos de datos numéricos"""
    for col, (min_val, max_val) in RANGES.items():
        if col in df.columns:
            valor = df[col].values[0]
            if not (min_val <= valor <= max_val):
                return {
                    'valido': False,
                    'mensaje': f'{col} debe estar entre {min_val} y {max_val}. Valor ingresado: {valor}'
                }
    return {'valido': True}

# ════════════════════════════════════════════════════════════════════════════
# MANEJO DE ERRORES
# ════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def no_encontrado(error):
    return jsonify({'error': 'Ruta no encontrada', 'status': 404}), 404

@app.errorhandler(500)
def error_servidor(error):
    return jsonify({'error': 'Error interno del servidor', 'status': 500}), 500

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Iniciando Baby Predictor en puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

