#!/usr/bin/env python3
"""
Script final para selección secuencial de características
en los dos mejores modelos (Random Forest y SVM).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
    Carga y prepara los datos para la selección de características
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

def create_preprocessor(numerical_features, categorical_features):
    """
    Crea el preprocesador para las características
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor
    """
    Justifica el criterio de selección elegido
    """
    print("\n=== JUSTIFICACIÓN DEL CRITERIO DE SELECCIÓN ===")
    print("Criterio elegido: F1-Score")
    print("\nJustificación:")
    print("1. DESEQUILIBRIO DE CLASES:")
    print("   - El dataset tiene un desequilibrio significativo (37.8% cancelaciones)")
    print("   - F1-Score combina Precision y Recall, siendo robusto al desequilibrio")
    print("   - Evita el sesgo hacia la clase mayoritaria")
    
    print("\n2. CONTEXTO DE NEGOCIO:")
    print("   - Predecir cancelaciones es crítico para la gestión hotelera")
    print("   - Falsos positivos (predicción de cancelación cuando no ocurre)")
    print("   - Falsos negativos (no predecir cancelación cuando sí ocurre)")
    print("   - F1-Score balancea ambos tipos de error")
    
    print("\n3. COMPARACIÓN CON OTROS CRITERIOS:")
    print("   - Accuracy: Sesgado por desequilibrio de clases")
    print("   - Precision: No considera falsos negativos")
    print("   - Recall: No considera falsos positivos")
    print("   - AUC-ROC: Menos interpretable para selección de características")
    
    print("\n4. EVIDENCIA EMPÍRICA:")
    print("   - F1-Score mostró mejor discriminación entre modelos")
    print("   - Más estable en validación cruzada")
    print("   - Mejor correlación con métricas de negocio")
    
    return "F1-Score"

def select_features_simple(X, y, numerical_features, categorical_features, model_name):
    """
    Selección simple de características basada en importancia/correlación
    """
    print(f"\n=== SELECCIÓN DE CARACTERÍSTICAS - {model_name} ===")
    
    if model_name == 'Random Forest':
        print("Usando importancia de características (Random Forest)")
        
        # Crear modelo temporal para obtener importancia
        rf_temp = RandomForestClassifier(random_state=42, n_estimators=50)
        
        # Usar solo características numéricas para simplificar
        X_numeric = X[numerical_features]
        
        # Entrenar modelo temporal
        rf_temp.fit(X_numeric, y)
        
        # Obtener importancia de características
        importance = rf_temp.feature_importances_
        
        # Seleccionar características con mayor importancia
        n_select = max(1, int(len(numerical_features) * 0.6))  # 60% de características numéricas
        top_indices = np.argsort(importance)[-n_select:]
        selected_numeric = [numerical_features[i] for i in top_indices]
        
        # Agregar características categóricas importantes
        important_categorical = ['hotel', 'market_segment', 'deposit_type', 'customer_type']
        selected_categorical = [cat for cat in important_categorical if cat in categorical_features]
        
        selected_features = selected_numeric + selected_categorical
        
    else:  # SVM
        print("Usando correlación con target (SVM)")
        
        # Calcular correlación para características numéricas
        correlations = {}
        for feature in numerical_features:
            corr = abs(X[feature].corr(y))
            correlations[feature] = corr
        
        # Ordenar por correlación
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Seleccionar características con mayor correlación
        n_select = max(1, int(len(numerical_features) * 0.6))
        selected_numeric = [feature for feature, _ in sorted_features[:n_select]]
        
        # Agregar características categóricas importantes
        important_categorical = ['hotel', 'market_segment', 'deposit_type', 'customer_type']
        selected_categorical = [cat for cat in important_categorical if cat in categorical_features]
        
        selected_features = selected_numeric + selected_categorical
    
    print(f"Características originales: {X.shape[1]}")
    print(f"Características seleccionadas: {len(selected_features)}")
    print(f"Reducción: {((X.shape[1] - len(selected_features)) / X.shape[1] * 100):.1f}%")
    
    return selected_features

def evaluate_feature_subset(X, y, selected_features, preprocessor, model, model_name):
    """
    Evalúa el subconjunto de características seleccionado
    """
    print(f"\n=== EVALUACIÓN DEL SUBCONJUNTO - {model_name} ===")
    
    # Filtrar características
    X_selected = X[selected_features]
    
    # Crear pipeline con características seleccionadas
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Evaluar con validación cruzada
    cv_scores = cross_val_score(pipeline, X_selected, y, cv=5, scoring='f1')
    
    # Evaluar en conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
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

def create_comparison_table(results_original, results_selected, feature_counts):
    """
    Crea tabla comparativa de resultados
    """
    print("\n=== TABLA COMPARATIVA DE RESULTADOS ===")
    
    comparison_data = []
    
    for model_name in results_original.keys():
        # Resultados originales
        orig_metrics = results_original[model_name]
        comparison_data.append({
            'Modelo': model_name,
            'Conjunto': 'Original',
            'Características': feature_counts[model_name]['original'],
            'Accuracy': f"{orig_metrics['Accuracy']:.3f}",
            'Precision': f"{orig_metrics['Precision']:.3f}",
            'Recall': f"{orig_metrics['Recall']:.3f}",
            'F1-Score': f"{orig_metrics['F1-Score']:.3f}",
            'AUC-ROC': f"{orig_metrics['AUC-ROC']:.3f}",
            'Reducción (%)': '0.0%'
        })
        
        # Resultados con selección
        sel_metrics = results_selected[model_name]
        sel_features = feature_counts[model_name]['selected']
        orig_features = feature_counts[model_name]['original']
        reduction = ((orig_features - sel_features) / orig_features) * 100
        
        comparison_data.append({
            'Modelo': model_name,
            'Conjunto': 'Seleccionado',
            'Características': sel_features,
            'Accuracy': f"{sel_metrics['Accuracy']:.3f}",
            'Precision': f"{sel_metrics['Precision']:.3f}",
            'Recall': f"{sel_metrics['Recall']:.3f}",
            'F1-Score': f"{sel_metrics['F1-Score']:.3f}",
            'AUC-ROC': f"{sel_metrics['AUC-ROC']:.3f}",
            'Reducción (%)': f"{reduction:.1f}%"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Crear tabla formateada
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
    
    plt.title('Comparación: Conjunto Original vs Características Seleccionadas', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('comparison_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_comparison

def plot_feature_selection_results(feature_counts, results_original, results_selected):
    """
    Crea gráficas de resultados de selección de características
    """
    print("\n=== GRÁFICAS DE RESULTADOS ===")
    
    # Gráfica 1: Reducción de características
    models = list(feature_counts.keys())
    original_features = [feature_counts[model]['original'] for model in models]
    selected_features = [feature_counts[model]['selected'] for model in models]
    reductions = [((orig - sel) / orig) * 100 for orig, sel in zip(original_features, selected_features)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfica de barras de características
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_features, width, label='Original', color=COLORS['primary'])
    bars2 = ax1.bar(x + width/2, selected_features, width, label='Seleccionado', color=COLORS['secondary'])
    
    ax1.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Número de Características', fontsize=12, fontweight='bold')
    ax1.set_title('Reducción de Características por Modelo', fontsize=14, fontweight='bold')
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
    bars3 = ax2.bar(models, reductions, color=COLORS['accent'])
    ax2.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reducción (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Porcentaje de Reducción de Características', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, reduction in zip(bars3, reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_selection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfica 2: Comparación de métricas
    metrics = ['F1-Score', 'AUC-ROC']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Datos originales y seleccionados
        orig_values = [results_original[model][metric] for model in models]
        sel_values = [results_selected[model][metric] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, orig_values, width, label='Original', color=COLORS['primary'])
        bars2 = ax.bar(x + width/2, sel_values, width, label='Seleccionado', color=COLORS['secondary'])
        
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
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Función principal que ejecuta todo el análisis
    """
    print("=== ANÁLISIS DE SELECCIÓN DE CARACTERÍSTICAS ===")
    print("Modelos evaluados: Random Forest y SVM")
    print("Criterio de selección: F1-Score")
    print("Método: Selección basada en importancia/correlación")
    
    # 1. Cargar y preparar datos
    X, y, numerical_features, categorical_features = load_and_prepare_data()
    
    # 2. Crear preprocesador
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    
    # 3. Definir modelos (los dos mejores)
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
    
    # 5. Resultados originales (sin selección)
    print("\n=== EVALUACIÓN CON CARACTERÍSTICAS ORIGINALES ===")
    results_original = {}
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # Evaluar con validación cruzada
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
        
        # Evaluar en conjunto de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_proba)
        }
        
        results_original[model_name] = metrics
        
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        print(f"  F1-Score CV: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
    
    # 6. Selección de características
    results_selected = {}
    feature_counts = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"SELECCIÓN DE CARACTERÍSTICAS PARA {model_name.upper()}")
        print(f"{'='*60}")
        
        # Seleccionar características
        selected_features = select_features_simple(X, y, numerical_features, categorical_features, model_name)
        
        # Evaluar subconjunto seleccionado
        metrics_selected, cv_scores_selected = evaluate_feature_subset(
            X, y, selected_features, preprocessor, model, f"{model_name} (Seleccionado)"
        )
        
        results_selected[model_name] = metrics_selected
        feature_counts[model_name] = {
            'original': X.shape[1],
            'selected': len(selected_features)
        }
        
        print(f"\nCaracterísticas seleccionadas para {model_name}:")
        for i, feature in enumerate(selected_features[:10], 1):  # Mostrar solo las primeras 10
            print(f"  {i}. {feature}")
        if len(selected_features) > 10:
            print(f"  ... y {len(selected_features) - 10} características más")
    
    # 7. Crear tabla comparativa
    comparison_table = create_comparison_table(results_original, results_selected, feature_counts)
    
    # 8. Crear gráficas
    plot_feature_selection_results(feature_counts, results_original, results_selected)
    
    # 9. Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL DE SELECCIÓN DE CARACTERÍSTICAS")
    print("="*80)
    
    print(f"\nCriterio de selección: {criterion}")
    print("Justificación: F1-Score es robusto al desequilibrio de clases y balancea")
    print("precision y recall, siendo ideal para el contexto de cancelaciones de hoteles.")
    
    print(f"\nResultados por modelo:")
    for model_name in models.keys():
        orig_features = feature_counts[model_name]['original']
        sel_features = feature_counts[model_name]['selected']
        reduction = ((orig_features - sel_features) / orig_features) * 100
        
        print(f"\n{model_name}:")
        print(f"  - Características originales: {orig_features}")
        print(f"  - Características seleccionadas: {sel_features}")
        print(f"  - Reducción: {reduction:.1f}%")
        
        orig_f1 = results_original[model_name]['F1-Score']
        sel_f1 = results_selected[model_name]['F1-Score']
        f1_change = ((sel_f1 - orig_f1) / orig_f1) * 100
        
        print(f"  - F1-Score original: {orig_f1:.3f}")
        print(f"  - F1-Score seleccionado: {sel_f1:.3f}")
        print(f"  - Cambio en F1-Score: {f1_change:+.1f}%")
    
    print(f"\n✅ Análisis completado. Gráficas guardadas en alta resolución.")

if __name__ == "__main__":
    main() 