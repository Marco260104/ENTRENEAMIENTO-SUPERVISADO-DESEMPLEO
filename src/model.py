import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)

def preparar_datos(df):
    X = df.drop('renuncia', axis=1)
    y = df['renuncia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X.columns

def evaluar_modelo(nombre, y_test, y_pred, proba, output_dir):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, proba)
    print(f"\n--- {nombre} ---")
    print(f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No renuncia', 'Renuncia']))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No','Sí'], yticklabels=['No','Sí'])
    plt.title(f'Matriz de confusión - {nombre}')
    plt.ylabel('Real'), plt.xlabel('Predicho')
    plt.savefig(os.path.join(output_dir, f'matriz_confusion_{nombre.lower().replace(" ","_")}.png'))
    plt.close()
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'AUC-ROC': auc}

def entrenar_modelos(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, features, output_dir):
    resultados = {}
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    proba_lr = lr.predict_proba(X_test_scaled)[:,1]
    res_lr = evaluar_modelo('Regresión Logística', y_test, y_pred_lr, proba_lr, output_dir)
    resultados['Regresión Logística'] = res_lr

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    proba_rf = rf.predict_proba(X_test)[:,1]
    res_rf = evaluar_modelo('Random Forest', y_test, y_pred_rf, proba_rf, output_dir)
    resultados['Random Forest'] = res_rf

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    proba_gb = gb.predict_proba(X_test)[:,1]
    res_gb = evaluar_modelo('Gradient Boosting', y_test, y_pred_gb, proba_gb, output_dir)
    resultados['Gradient Boosting'] = res_gb

    plt.figure(figsize=(8,6))
    fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, proba_gb)
    plt.plot(fpr_lr, tpr_lr, label=f'RegLog (AUC={res_lr["AUC-ROC"]:.3f})')
    plt.plot(fpr_rf, tpr_rf, label=f'RandForest (AUC={res_rf["AUC-ROC"]:.3f})')
    plt.plot(fpr_gb, tpr_gb, label=f'GradBoost (AUC={res_gb["AUC-ROC"]:.3f})')
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.xlabel('Tasa Falsos Positivos'), plt.ylabel('Tasa Verdaderos Positivos')
    plt.title('Curvas ROC comparativas')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'curvas_roc_comparativas.png'))
    plt.close()

    importances_rf = rf.feature_importances_
    idx_rf = np.argsort(importances_rf)[::-1]
    importances_gb = gb.feature_importances_
    idx_gb = np.argsort(importances_gb)[::-1]
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    axes[0].bar(range(len(importances_rf)), importances_rf[idx_rf], color='skyblue')
    axes[0].set_xticks(range(len(importances_rf)))
    axes[0].set_xticklabels(features[idx_rf], rotation=45, ha='right')
    axes[0].set_title('Feature Importance - Random Forest')
    axes[1].bar(range(len(importances_gb)), importances_gb[idx_gb], color='lightgreen')
    axes[1].set_xticks(range(len(importances_gb)))
    axes[1].set_xticklabels(features[idx_gb], rotation=45, ha='right')
    axes[1].set_title('Feature Importance - Gradient Boosting')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    return resultados, rf, gb, proba_rf, proba_gb, proba_lr

def cross_validation_mejor_modelo(X_train, y_train, modelo, cv=5):
    scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='f1')
    print(f"\nCV F1 ({cv}-fold): Media={scores.mean():.4f}, Std={scores.std():.4f}")

def optimizar_hiperparametros(X_train, y_train, X_test, y_test, output_dir):
    # Definimos la grilla de hiperparámetros para RandomForest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_base = RandomForestClassifier(random_state=42)
    
    # Aplicamos GridSearchCV para una búsqueda exhaustiva
    grid_search = GridSearchCV(rf_base, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\nMejores hiperparámetros (GridSearchCV):", grid_search.best_params_)
    print(f"Mejor F1 (validación cruzada): {grid_search.best_score_:.4f}")

    rf_opt = grid_search.best_estimator_
    y_pred_opt = rf_opt.predict(X_test)
    f1_opt = f1_score(y_test, y_pred_opt)
    auc_opt = roc_auc_score(y_test, rf_opt.predict_proba(X_test)[:,1])
    print(f"Optimizado en test -> F1: {f1_opt:.4f}, AUC: {auc_opt:.4f}")

    rf_base.fit(X_train, y_train)
    f1_base = f1_score(y_test, rf_base.predict(X_test))
    auc_base = roc_auc_score(y_test, rf_base.predict_proba(X_test)[:,1])
    comparacion = pd.DataFrame({
        'Modelo': ['Random Forest Base', 'Random Forest Optimizado'],
        'F1-Score': [f1_base, f1_opt],
        'AUC-ROC': [auc_base, auc_opt]
    })
    print("\nComparación base vs optimizado:")
    print(comparacion)

    cm_opt = confusion_matrix(y_test, y_pred_opt)
    plt.figure()
    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', xticklabels=['No','Sí'], yticklabels=['No','Sí'])
    plt.title('Matriz de Confusión - Random Forest Optimizado')
    plt.ylabel('Real'), plt.xlabel('Predicho')
    plt.savefig(os.path.join(output_dir, 'matriz_confusion_optimizado.png'))
    plt.close()
    return rf_opt
