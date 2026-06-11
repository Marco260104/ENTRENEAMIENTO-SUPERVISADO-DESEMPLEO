# 🤖 Sistema de Predicción de Renuncias — Aprendizaje Supervisado

> **Tarea Complementaria · Ingeniería de Software · 7mo Semestre**  
> Estudiante: Marco Torres  
> Docente: Isaac Torres

Sistema de Machine Learning que predice si un empleado **renunciará en los próximos 6 meses**, basado en características recopiladas durante su permanencia en la empresa. Incluye generación de dataset sintético con lógica coherente, análisis exploratorio completo, comparación de 3 modelos de clasificación y optimización de hiperparámetros.

---

## 📁 Estructura del proyecto

```
AprendizajeSupervisado/
│
├── src/
│   ├── data_generation.py   # Generación del dataset sintético con lógica coherente
│   ├── eda.py               # Análisis exploratorio (EDA) y visualizaciones
│   └── model.py             # Entrenamiento, evaluación y optimización de modelos
│
├── data/
│   ├── dataset_renuncias.csv    # Dataset generado (350 registros)
│   ├── scaler.joblib            # StandardScaler ajustado (para la app)
│   └── modelo_rf_opt.joblib     # Modelo Random Forest optimizado (para la app)
│
├── figures/                     # Gráficos generados automáticamente
│   ├── heatmap_correlacion.png
│   ├── distribucion_renuncia.png
│   ├── boxplots_comparativos.png
│   ├── curvas_roc_comparativas.png
│   ├── feature_importance.png
│   ├── matriz_confusion_regresión_logística.png
│   ├── matriz_confusion_random_forest.png
│   ├── matriz_confusion_gradient_boosting.png
│   └── matriz_confusion_optimizado.png
│
├── main.py                  # Pipeline principal (ejecutar este archivo)
├── app.py                   # Interfaz web interactiva (Flask)
├── requirements.txt         # Dependencias del proyecto
└── README.md
```

---

## ⚙️ Requisitos previos

- Python **3.9 o superior**
- pip actualizado

---

## 🚀 Instalación y ejecución

### 1. Clonar o descargar el repositorio

```bash
# Si lo clonaste con git
git clone <url-del-repositorio>
cd AprendizajeSupervisado
```

### 2. Crear y activar el entorno virtual

```powershell
# Crear entorno virtual
python -m venv venv

# Activar en Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activar en Windows (CMD)
venv\Scripts\activate.bat
```

### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 4. Ejecutar el pipeline principal

```powershell
python main.py
```

Esto ejecutará en orden:
1. ✅ Generación del dataset (350 registros → `data/dataset_renuncias.csv`)
2. ✅ Análisis exploratorio (gráficos → `figures/`)
3. ✅ Entrenamiento de 3 modelos (Regresión Logística, Random Forest, Gradient Boosting)
4. ✅ Cross-validation k=5 sobre el mejor modelo
5. ✅ Optimización de hiperparámetros con GridSearchCV
6. ✅ Guardado del scaler y modelo optimizado en `data/`

### 5. (Opcional) Lanzar la interfaz web

```powershell
python app.py
```

Luego abre el navegador en: `http://127.0.0.1:5000`

---

## 🧠 Modelos implementados

| Modelo | Descripción |
|---|---|
| **Regresión Logística** | Modelo base lineal, requiere estandarización de features |
| **Random Forest** | Ensamble de árboles de decisión, robusto y escalable |
| **Gradient Boosting** | Boosting secuencial, alto poder predictivo |

El mejor modelo (**Random Forest**) se optimiza con `GridSearchCV` evaluando:
- `n_estimators`: [50, 100, 150]
- `max_depth`: [None, 5, 10]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

---

## 📊 Variables del dataset

| Variable | Tipo | Rango |
|---|---|---|
| `edad` | Entero | 22–60 |
| `años_en_empresa` | Entero | 0–20 |
| `salario_mensual` | Flotante | 500–5000 USD |
| `horas_extra_semana` | Entero | 0–20 |
| `satisfaccion_laboral` | Entero | 1–5 |
| `num_proyectos_año` | Entero | 1–10 |
| `distancia_casa_trabajo_km` | Entero | 1–80 |
| `ultima_evaluacion_desempeño` | Flotante | 0.0–1.0 |
| `capacitaciones_recibidas` | Entero | 0–5 |
| `tiene_ascenso_ultimos_2_años` | Binario | 0 / 1 |
| **`renuncia`** *(objetivo)* | Binario | 0 = se queda · 1 = renuncia |

---

## 📈 Gráficos generados

| Archivo | Contenido |
|---|---|
| `heatmap_correlacion.png` | Correlación entre todas las variables |
| `distribucion_renuncia.png` | Balance de clases de la variable objetivo |
| `boxplots_comparativos.png` | Salario y satisfacción separados por renuncia |
| `curvas_roc_comparativas.png` | Curvas ROC de los 3 modelos en una figura |
| `feature_importance.png` | Importancia de variables (RF y GB) |
| `matriz_confusion_*.png` | Matrices de confusión de cada modelo |

---

## 🛠️ Dependencias

```
pandas
numpy
matplotlib
seaborn
scikit-learn
Faker
```

---

## 📝 Notas

- El dataset se genera con `random_state=42` para garantizar **reproducibilidad** total.
- El `StandardScaler` se ajusta **únicamente sobre el conjunto de entrenamiento** para evitar data leakage.
- Todos los gráficos se guardan automáticamente en la carpeta `figures/` al ejecutar `main.py`.
