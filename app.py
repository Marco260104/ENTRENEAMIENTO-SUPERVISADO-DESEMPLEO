import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Cargar el modelo entrenado
MODEL_PATH = 'data/modelo_rf_opt.joblib'
SCALER_PATH = 'data/scaler.joblib'

model = None
scaler = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'El modelo no está cargado. Ejecuta main.py primero.'}), 500
    
    try:
        data = request.json
        
        # Validar y extraer características en el orden correcto
        features = [
            'edad',
            'años_en_empresa',
            'salario_mensual',
            'horas_extra_semana',
            'satisfaccion_laboral',
            'num_proyectos_año',
            'distancia_casa_trabajo_km',
            'ultima_evaluacion_desempeño',
            'capacitaciones_recibidas',
            'tiene_ascenso_ultimos_2_años'
        ]
        
        # Crear DataFrame de una fila
        input_data = {}
        for f in features:
            if f not in data:
                return jsonify({'error': f'Falta la característica requerida: {f}'}), 400
            
            # Convertir a tipos numéricos apropiados
            if f in ['edad', 'años_en_empresa', 'satisfaccion_laboral', 'num_proyectos_año', 'capacitaciones_recibidas', 'tiene_ascenso_ultimos_2_años']:
                input_data[f] = int(data[f])
            else:
                input_data[f] = float(data[f])
        
        df_pred = pd.DataFrame([input_data], columns=features)
        
        # Obtener predicción y probabilidades
        pred = int(model.predict(df_pred)[0])
        prob = float(model.predict_proba(df_pred)[0][1])  # Probabilidad de clase 1 (renuncia)
        
        # Generar recomendaciones dinámicas basadas en las variables que aumentan la renuncia
        recomendaciones = []
        if input_data['satisfaccion_laboral'] <= 2:
            recomendaciones.append("Implementar plan de mejora de clima laboral o entrevista de satisfacción. Su satisfacción actual es muy baja.")
        if input_data['horas_extra_semana'] > 10:
            recomendaciones.append("Reducir la carga de horas extra (actualmente trabaja más de 10 horas extra por semana).")
        if input_data['tiene_ascenso_ultimos_2_años'] == 0 and input_data['años_en_empresa'] >= 3:
            recomendaciones.append("Revisar plan de carrera y elegibilidad de ascenso (lleva más de 3 años en la empresa sin ascenso).")
        if input_data['salario_mensual'] < 1500 and input_data['ultima_evaluacion_desempeño'] > 0.8:
            recomendaciones.append("Ajustar el salario competitivo de este empleado (alto desempeño con salario por debajo del promedio).")
        if input_data['distancia_casa_trabajo_km'] > 30:
            recomendaciones.append("Ofrecer modalidades de teletrabajo o ayuda de transporte por la alta distancia a la oficina.")
        
        if not recomendaciones:
            recomendaciones.append("Mantener las condiciones laborales actuales. El perfil muestra alta estabilidad laboral.")
            
        return jsonify({
            'renuncia': pred,
            'probabilidad_renuncia': prob,
            'riesgo': 'Alto' if prob >= 0.5 else ('Medio' if prob >= 0.25 else 'Bajo'),
            'recomendaciones': recomendaciones
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ejecutar en puerto local
    app.run(debug=True, port=5002)
