import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def realizar_eda(df, output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)

    print("=== Estadísticas descriptivas ===")
    print(df.describe(), "\n")

    plt.figure(figsize=(12,8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap de correlación')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_correlacion.png'))
    plt.close()

    print("Correlaciones con renuncia:")
    print(corr['renuncia'].sort_values(ascending=False), "\n")

    plt.figure()
    ax = sns.countplot(x='renuncia', data=df, hue='renuncia', palette='Set2', legend=False)
    plt.title('Distribución de renuncia')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2., p.get_height()),
                    ha='center', va='center', xytext=(0,5), textcoords='offset points')
    plt.savefig(os.path.join(output_dir, 'distribucion_renuncia.png'))
    plt.close()
    pct = df['renuncia'].mean()*100
    print(f"Porcentaje de renuncias: {pct:.1f}%")
    if pct < 40 or pct > 60:
        print("Dataset desbalanceado -> se usarán F1 y AUC-ROC.\n")

    fig, axes = plt.subplots(1,2,figsize=(14,5))
    sns.boxplot(x='renuncia', y='salario_mensual', data=df, ax=axes[0], hue='renuncia', palette='Set2', legend=False)
    axes[0].set_title('Salario mensual según renuncia')
    sns.boxplot(x='renuncia', y='satisfaccion_laboral', data=df, ax=axes[1], hue='renuncia', palette='Set2', legend=False)
    axes[1].set_title('Satisfacción laboral según renuncia')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplots_comparativos.png'))
    plt.close()
