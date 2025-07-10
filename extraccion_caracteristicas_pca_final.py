#!/usr/bin/env python3
"""
Script final y eficiente para extracción de características usando PCA
en los dos mejores modelos (Random Forest y SVM).
Optimizado para manejar alta dimensionalidad del dataset de hoteles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colores para las gráficas
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#6C757D'
}

def load_and_prepare_data():
    """
    Carga y prepara los datos para el análisis PCA
    """
    print("=== CARGA Y PREPARACIÓN DE DATOS ===")
    
    # Cargar datos
    df = pd.read_csv('datos/hotel_booking.csv')
    
    # Reducir dataset para eficiencia
    df = df.sample(frac=0.1, random_state=42)
    
    # Limpieza básica
    df['children'] = df['children'].fillna(0)
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    df = df[df['adr'] >= 0]
    df = df[df['adults'] + df['children'] + df['babies'] > 0]
    df = df.drop(columns=['reservation_status', 'reservation_status_date'])
    
    # Separar features y target
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    # Identificar características numéricas y categóricas
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Características numéricas: {len(numerical_features)}")
    print(f"Características categóricas: {len(categorical_features)}")
    print(f"Total de características: {X.shape[1]}")
    
    return X, y, numerical_features, categorical_features

def create_efficient_preprocessor(numerical_features, categorical_features):
    """
    Crea un preprocesador eficiente que limita las características categóricas
    """
    # Limitar características categóricas para evitar alta dimensionalidad
    limited_categorical = ['hotel', 'market_segment', 'deposit_type', 'customer_type']
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, limited_categorical)
        ],
        remainder='drop'  # Eliminar otras características categóricas
    )
    
    return preprocessor, limited_categorical
    """
    Justifica el criterio de selección de componentes principales
    """
    print("\n=== JUSTIFICACIÓN DEL CRITERIO DE SELECCIÓN DE COMPONENTES ===")
    print("Criterio elegido: 95% de Varianza Explicada Acumulada")
    print("\nJustificación:")
    print("1. CONSERVACIÓN DE INFORMACIÓN:")
    print("   - 95% de varianza asegura retener la mayoría de la información")
    print("   - Balance entre reducción dimensional y preservación de datos")
    print("   - Estándar ampliamente aceptado en la literatura")
    
    print("\n2. REDUCCIÓN DIMENSIONAL EFECTIVA:")
    print("   - Elimina redundancia manteniendo características discriminativas")
    print("   - Reduce overfitting al eliminar ruido y correlaciones")
    print("   - Mejora eficiencia computacional")
    
    print("\n3. COMPARACIÓN CON OTROS CRITERIOS:")
    print("   - Kaiser (eigenvalues > 1): Puede ser muy conservador")
    print("   - Scree plot: Subjetivo y difícil de automatizar")
    print("   - 99% varianza: Demasiado conservador, poca reducción")
    print("   - 90% varianza: Podría perder información importante")
    
    print("\n4. EVIDENCIA EMPÍRICA:")
    print("   - 95% es óptimo para datasets con ruido moderado")
    print("   - Mantiene capacidad predictiva en la mayoría de casos")
    print("   - Permite reducción significativa sin pérdida crítica")
    
    return "95% Varianza Explicada"

def analyze_pca_components_efficient(X, y, preprocessor):
    """
    Analiza los componentes principales de forma eficiente
    """
    print("\n=== ANÁLISIS DE COMPONENTES PRINCIPALES ===")
    
    # Aplicar preprocesamiento
    X_preprocessed = preprocessor.fit_transform(X)
    print(f"Dimensiones después del preprocesamiento: {X_preprocessed.shape}")
    
    # Limitar el número de componentes para análisis
    max_components = min(100, X_preprocessed.shape[0] - 1, X_preprocessed.shape[1])
    
    # Aplicar PCA
    pca_full = PCA(n_components=max_components, random_state=42)
    X_pca_full = pca_full.fit_transform(X_preprocessed)
    
    # Calcular varianza explicada acumulada
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Encontrar número de componentes para diferentes umbrales
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1 if np.any(cumulative_variance >= 0.99) else max_components
    
    print(f"Dimensiones originales (después preprocesamiento): {X_preprocessed.shape[1]}")
    print(f"Componentes para 90% varianza: {n_components_90}")
    print(f"Componentes para 95% varianza: {n_components_95}")
    print(f"Componentes para 99% varianza: {n_components_99}")
    
    # Crear gráfica de varianza explicada
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Varianza explicada por componente
    plt.subplot(2, 3, 1)
    components_to_show = min(30, len(pca_full.explained_variance_ratio_))
    plt.plot(range(1, components_to_show + 1), 
             pca_full.explained_variance_ratio_[:components_to_show], 'bo-', linewidth=2, markersize=4)
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Varianza explicada acumulada
    plt.subplot(2, 3, 2)
    plt.plot(range(1, components_to_show + 1), 
             cumulative_variance[:components_to_show], 'ro-', linewidth=2, markersize=4)
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90%')
    plt.axhline(y=0.95, color='green', linestyle='--', label='95%')
    plt.axhline(y=0.99, color='red', linestyle='--', label='99%')
    plt.axvline(x=n_components_95, color='green', linestyle=':', alpha=0.7)
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada Acumulada')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Comparación de umbrales
    plt.subplot(2, 3, 3)
    thresholds = [90, 95, 99]
    components = [n_components_90, n_components_95, n_components_99]
    reduction = [(X_preprocessed.shape[1] - comp) / X_preprocessed.shape[1] * 100 for comp in components]
    
    bars = plt.bar(range(len(thresholds)), components, 
                  color=[COLORS['accent'], COLORS['primary'], COLORS['secondary']])
    plt.xlabel('Umbral de Varianza (%)')
    plt.ylabel('Número de Componentes')
    plt.title('Componentes por Umbral de Varianza')
    plt.xticks(range(len(thresholds)), [f'{t}%' for t in thresholds])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, comp, red in zip(bars, components, reduction):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{comp}\n({red:.1f}% red.)', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Primeros componentes principales
    plt.subplot(2, 3, 4)
    first_15_variance = pca_full.explained_variance_ratio_[:15]
    plt.bar(range(1, len(first_15_variance) + 1), first_15_variance, color=COLORS['neutral'])
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Primeros 15 Componentes Principales')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 5: Scree plot
    plt.subplot(2, 3, 5)
    eigenvalues = pca_full.explained_variance_[:20]
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'go-', linewidth=2, markersize=6)
    plt.axhline(y=1, color='red', linestyle='--', label='Kaiser Criterion (λ=1)')
    plt.xlabel('Componente Principal')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot (Primeros 20 componentes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 6: Distribución de varianza acumulada
    plt.subplot(2, 3, 6)
    milestones = [5, 10, 15, 20, 30]
    milestone_variance = []
    for milestone in milestones:
        if milestone <= len(cumulative_variance):
            milestone_variance.append(cumulative_variance[milestone-1])
        else:
            milestone_variance.append(cumulative_variance[-1])
    
    plt.plot(milestones, milestone_variance, 'mo-', linewidth=2, markersize=8)
    plt.axhline(y=0.95, color='green', linestyle='--', label='95% objetivo')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Acumulada en Hitos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return n_components_95, X_preprocessed.shape[1], pca_full.explained_variance_ratio_

def evaluate_model_with_pca(X, y, preprocessor, model, model_name, n_components):
    """
    Evalúa un modelo usando PCA
    """
    print(f"\n=== EVALUACIÓN CON PCA - {model_name} ===")
    print(f"Usando {n_components} componentes principales")
    
    # Crear pipeline con PCA
    pipeline_pca = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Evaluar con validación cruzada
    cv_scores = cross_val_score(pipeline_pca, X, y, cv=5, scoring='f1')
    
    # Evaluar en conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline_pca.fit(X_train, y_train)
    y_pred = pipeline_pca.predict(X_test)
    y_proba = pipeline_pca.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    print(f"Resultados en test set:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"F1-Score CV (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
    
    return metrics, cv_scores

def evaluate_model_original(X, y, preprocessor, model, model_name):
    """
    Evalúa un modelo sin PCA
    """
    print(f"\n=== EVALUACIÓN SIN PCA - {model_name} ===")
    
    # Crear pipeline sin PCA
    pipeline_original = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Evaluar con validación cruzada
    cv_scores = cross_val_score(pipeline_original, X, y, cv=5, scoring='f1')
    
    # Evaluar en conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline_original.fit(X_train, y_train)
    y_pred = pipeline_original.predict(X_test)
    y_proba = pipeline_original.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba)
    }
    
    print(f"Resultados en test set:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"F1-Score CV (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
    
    return metrics, cv_scores

def create_pca_comparison_table(results_original, results_pca, n_components, original_features):
    """
    Crea tabla comparativa de resultados con y sin PCA
    """
    print("\n=== TABLA COMPARATIVA DE RESULTADOS PCA ===")
    
    comparison_data = []
    
    for model_name in results_original.keys():
        # Resultados originales
        orig_metrics = results_original[model_name]
        comparison_data.append({
            'Modelo': model_name,
            'Método': 'Original',
            'Características': original_features,
            'Accuracy': f"{orig_metrics['Accuracy']:.3f}",
            'Precision': f"{orig_metrics['Precision']:.3f}",
            'Recall': f"{orig_metrics['Recall']:.3f}",
            'F1-Score': f"{orig_metrics['F1-Score']:.3f}",
            'AUC-ROC': f"{orig_metrics['AUC-ROC']:.3f}",
            'Reducción (%)': '0.0%'
        })
        
        # Resultados con PCA
        pca_metrics = results_pca[model_name]
        reduction = ((original_features - n_components) / original_features) * 100
        
        comparison_data.append({
            'Modelo': model_name,
            'Método': 'PCA',
            'Características': n_components,
            'Accuracy': f"{pca_metrics['Accuracy']:.3f}",
            'Precision': f"{pca_metrics['Precision']:.3f}",
            'Recall': f"{pca_metrics['Recall']:.3f}",
            'F1-Score': f"{pca_metrics['F1-Score']:.3f}",
            'AUC-ROC': f"{pca_metrics['AUC-ROC']:.3f}",
            'Reducción (%)': f"{reduction:.1f}%"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Mostrar tabla en consola
    print("\nTabla Comparativa de Resultados:")
    print("=" * 100)
    print(df_comparison.to_string(index=False))
    
    # Crear tabla formateada como imagen
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_comparison.values.tolist(), 
                    colLabels=df_comparison.columns.tolist(), 
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Formatear tabla
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Colorear encabezados
    for i in range(len(df_comparison.columns)):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear filas alternas
    for i in range(1, len(df_comparison) + 1):
        for j in range(len(df_comparison.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Comparación: Características Originales vs PCA', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('pca_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_comparison

def plot_pca_results(results_original, results_pca, n_components, original_features):
    """
    Crea gráficas de resultados de PCA
    """
    print("\n=== GRÁFICAS DE RESULTADOS PCA ===")
    
    models = list(results_original.keys())
    
    # Gráfica 1: Reducción de dimensionalidad
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfica de barras de características
    x = np.arange(len(models))
    width = 0.35
    
    original_dims = [original_features] * len(models)
    pca_dims = [n_components] * len(models)
    
    bars1 = ax1.bar(x - width/2, original_dims, width, label='Original', color=COLORS['primary'])
    bars2 = ax1.bar(x + width/2, pca_dims, width, label='PCA', color=COLORS['secondary'])
    
    ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Número de Características', fontsize=12, fontweight='bold')
    ax1.set_title('Reducción Dimensional con PCA', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfica de porcentaje de reducción
    reduction_pct = ((original_features - n_components) / original_features) * 100
    reductions = [reduction_pct] * len(models)
    
    bars3 = ax2.bar(models, reductions, color=COLORS['accent'])
    ax2.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reducción (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Porcentaje de Reducción Dimensional', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, reduction in zip(bars3, reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pca_reduction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfica 2: Comparación de métricas
    metrics = ['F1-Score', 'AUC-ROC']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Datos originales y PCA
        orig_values = [results_original[model][metric] for model in models]
        pca_values = [results_pca[model][metric] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, orig_values, width, label='Original', color=COLORS['primary'])
        bars2 = ax.bar(x + width/2, pca_values, width, label='PCA', color=COLORS['secondary'])
        
        ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Comparación de {metric}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('pca_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Función principal que ejecuta todo el análisis PCA
    """
    print("=== ANÁLISIS DE EXTRACCIÓN DE CARACTERÍSTICAS CON PCA ===")
    print("Modelos evaluados: Random Forest y SVM")
    print("Método: Análisis de Componentes Principales (PCA)")
    
    # 1. Cargar y preparar datos
    X, y, numerical_features, categorical_features = load_and_prepare_data()
    
    # 2. Crear preprocesador eficiente
    preprocessor, used_categorical = create_efficient_preprocessor(numerical_features, categorical_features)
    print(f"\nCaracterísticas categóricas utilizadas: {used_categorical}")

    # 3. Analizar componentes principales
    n_components, original_features, explained_variance = analyze_pca_components_efficient(X, y, preprocessor)
    
    print(f"\n🎯 Componentes seleccionados: {n_components}")
    print(f"🎯 Características originales (post-preprocesamiento): {original_features}")
    print(f"🎯 Reducción: {((original_features - n_components) / original_features * 100):.1f}%")
    print(f"🎯 Varianza explicada: {explained_variance.sum():.1%}")
    
    # 5. Definir modelos (los dos mejores)
    models = {
        'Random Forest': RandomForestClassifier(
            random_state=42, 
            n_estimators=100, 
            max_depth=30, 
            max_features='sqrt'
        ),
        'SVM': SVC(
            random_state=42, 
            probability=True, 
            C=10, 
            kernel='rbf', 
            gamma='scale'
        )
    }
    
    # 6. Evaluación con características originales
    print("\n" + "="*60)
    print("EVALUACIÓN CON CARACTERÍSTICAS ORIGINALES")
    print("="*60)
    
    results_original = {}
    for model_name, model in models.items():
        metrics, cv_scores = evaluate_model_original(X, y, preprocessor, model, model_name)
        results_original[model_name] = metrics
    
    # 7. Evaluación con PCA
    print("\n" + "="*60)
    print("EVALUACIÓN CON PCA")
    print("="*60)
    
    results_pca = {}
    for model_name, model in models.items():
        metrics, cv_scores = evaluate_model_with_pca(X, y, preprocessor, model, model_name, n_components)
        results_pca[model_name] = metrics
    
    # 8. Crear tabla comparativa
    comparison_table = create_pca_comparison_table(results_original, results_pca, n_components, original_features)
    
    # 9. Crear gráficas
    plot_pca_results(results_original, results_pca, n_components, original_features)
    
    # 10. Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL DE EXTRACCIÓN DE CARACTERÍSTICAS PCA")
    print("="*80)
    
    print(f"\nCriterio de selección: {criterion}")
    print("Justificación: 95% de varianza explicada es el estándar para mantener")
    print("información crítica mientras se logra reducción dimensional significativa.")
    
    print(f"\nReducción dimensional:")
    print(f"  - Características originales (post-preprocesamiento): {original_features}")
    print(f"  - Componentes principales: {n_components}")
    print(f"  - Reducción: {((original_features - n_components) / original_features * 100):.1f}%")
    print(f"  - Varianza explicada: {explained_variance.sum():.1%}")
    
    print(f"\nResultados por modelo:")
    for model_name in models.keys():
        orig_f1 = results_original[model_name]['F1-Score']
        pca_f1 = results_pca[model_name]['F1-Score']
        f1_change = ((pca_f1 - orig_f1) / orig_f1) * 100
        
        orig_auc = results_original[model_name]['AUC-ROC']
        pca_auc = results_pca[model_name]['AUC-ROC']
        auc_change = ((pca_auc - orig_auc) / orig_auc) * 100
        
        print(f"\n{model_name}:")
        print(f"  - F1-Score original: {orig_f1:.3f}")
        print(f"  - F1-Score PCA: {pca_f1:.3f}")
        print(f"  - Cambio en F1-Score: {f1_change:+.1f}%")
        print(f"  - AUC-ROC original: {orig_auc:.3f}")
        print(f"  - AUC-ROC PCA: {pca_auc:.3f}")
        print(f"  - Cambio en AUC-ROC: {auc_change:+.1f}%")
    
    print(f"\n✅ Análisis PCA completado. Gráficas guardadas en alta resolución.")

if __name__ == "__main__":
    main() 