import os
import pandas as pd
from src.data_generation import generar_dataset
from src.eda import realizar_eda
from src.model import preparar_datos, entrenar_modelos, cross_validation_mejor_modelo, optimizar_hiperparametros

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print("===== 1. GENERANDO DATASET =====")
    df = generar_dataset()
    df.to_csv('data/dataset_renuncias.csv', index=False)
    print(f"Dataset: {df.shape[0]} registros\n")

    print("===== 2. EDA =====")
    realizar_eda(df, output_dir='figures')

    print("===== 3. PREPARACIÓN DE DATOS =====")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, features = preparar_datos(df)

    print("===== 4. MODELADO =====")
    resultados, rf, gb, proba_rf, proba_gb, proba_lr = entrenar_modelos(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, features, 'figures')
    print("\n===== TABLA COMPARATIVA =====")
    print(pd.DataFrame(resultados).T)

    print("\n===== 5. CROSS-VALIDATION (Random Forest) =====")
    cross_validation_mejor_modelo(X_train, y_train, rf, cv=5)

    print("\n===== 6. OPTIMIZACIÓN (Random Forest) =====")
    rf_opt = optimizar_hiperparametros(X_train, y_train, X_test, y_test, 'figures')

    # Guardar scaler y modelo optimizado para la interfaz interactiva
    import joblib
    joblib.dump(scaler, 'data/scaler.joblib')
    joblib.dump(rf_opt, 'data/modelo_rf_opt.joblib')
    print("\nScaler y Modelo optimizado guardados en la carpeta 'data/'")

    print("\n===== PROCESO COMPLETO =====")

if __name__ == "__main__":
    main()
