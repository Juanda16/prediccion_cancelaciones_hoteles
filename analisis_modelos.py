# ---------------------------------------------
# PROYECTO: Predicción de Cancelaciones Hoteleras
# ---------------------------------------------

# Importación de librerías necesarias
import os  # Para operaciones del sistema de archivos
import zipfile  # Para manejar archivos comprimidos ZIP
import pandas as pd  # Para manipulación de datos en DataFrames
import numpy as np  # Para operaciones numéricas y manejo de NaN
from sklearn.model_selection import StratifiedKFold, GridSearchCV  # Para validación cruzada y búsqueda de hiperparámetros
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Para normalización y codificación de variables categóricas
from sklearn.compose import ColumnTransformer  # Para aplicar transformaciones a columnas específicas
from sklearn.pipeline import Pipeline  # Para crear flujos de procesamiento
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # Para evaluar modelos
from imblearn.over_sampling import SMOTE  # Para balancear clases usando sobremuestreo sintético
from imblearn.pipeline import Pipeline as ImbPipeline  # Pipeline compatible con imbalanced-learn

# -----------------------------
# 1. Carga de los datos
# -----------------------------
file_name = 'datos/hotel_booking.csv'  # Nombre del archivo de datos
try:
    # Intentamos leer el archivo CSV en un DataFrame de pandas
    fetched_df = pd.read_csv(file_name)
    print(f"\n¡Dataset '{os.path.basename(file_name)}' cargado exitosamente en DataFrame!\n")
except FileNotFoundError:
    # Si no se encuentra el archivo, mostramos un error y detenemos la ejecución
    print(f"Error: No se encontró el archivo CSV en {file_name} después de descomprimir. Verifique el contenido descomprimido.")
    exit()

# Mostramos información inicial del DataFrame para entender los tipos de datos y valores nulos
print("Información inicial del DataFrame:")
fetched_df.info()
print("\n")

# -----------------------------
# 2. Reducción del tamaño del dataset (opcional)
# -----------------------------
# Esto es útil para evitar problemas de memoria en entornos limitados
percentage_to_keep = 0.1  # Porcentaje de filas a conservar (10%)

if percentage_to_keep < 1.0:
    print(f"Reduciendo el tamaño del dataset al {percentage_to_keep*100:.2f}% de las filas originales.")
    df = fetched_df.sample(frac=percentage_to_keep, random_state=42)  # Muestreo aleatorio reproducible
    print(f"Nuevo tamaño del dataset: {df.shape[0]} filas.")
else:
    df = fetched_df.copy()

# Separamos las variables predictoras (X) y la variable objetivo (y)
X = df.drop('is_canceled', axis=1)  # X contiene todas las columnas excepto 'is_canceled'
y = df['is_canceled']  # y contiene la columna objetivo

print("\nDataFrame después de la posible reducción de tamaño:")
print(X.head())
print(f"Distribución de la variable objetivo tras la reducción:\n{y.value_counts(normalize=True)}\n")

# Identificamos las variables numéricas y categóricas para el preprocesamiento posterior
numerical_features = X.select_dtypes(include=np.number).columns.tolist()  # Lista de columnas numéricas
categorical_features = X.select_dtypes(include='object').columns.tolist()  # Lista de columnas categóricas

print(f"Características numéricas identificadas: {numerical_features}")
print(f"Características categóricas identificadas: {categorical_features}\n")

# -----------------------------
# 3. Limpieza y preprocesamiento inicial
# -----------------------------
# Rellenamos valores nulos en columnas relevantes
# Se asume que valores nulos en 'children', 'agent' y 'company' pueden ser tratados como 0
# Esto es razonable porque la ausencia de estos datos suele indicar que no aplica (ej: sin niños, sin agente, sin compañía)
df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

# Eliminamos filas con valores de ADR (tarifa diaria promedio) negativos o cero, ya que suelen ser errores de registro
df = df[df['adr'] >= 0]
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Reemplazamos infinitos por NaN si existen
df.dropna(subset=['adr'], inplace=True)  # Eliminamos filas donde 'adr' sea NaN

# Eliminamos filas donde no hay huéspedes (adultos, niños y bebés todos en cero)
initial_rows = df.shape[0]
df = df[df['adults'] + df['children'] + df['babies'] > 0]
print(f"Se eliminaron {initial_rows - df.shape[0]} filas con 0 huéspedes en total.\n")

# Eliminamos columnas que pueden causar fuga de información (leakage) hacia la predicción
df = df.drop(columns=['reservation_status', 'reservation_status_date'])  # Estas columnas indican directamente la cancelación

# Volvemos a separar X e y tras la limpieza
df = df.reset_index(drop=True)
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

print("DataFrame tras limpieza y preprocesamiento inicial:")
print(X.head())
print(f"Distribución de la variable objetivo:\n{y.value_counts(normalize=True)}\n")

# Reidentificamos las variables numéricas y categóricas tras la limpieza
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Características numéricas identificadas: {numerical_features}")
print(f"Características categóricas identificadas: {categorical_features}\n")

# -----------------------------
# 4. Definición del pipeline de preprocesamiento
# -----------------------------

# Creamos el transformador para variables numéricas: escalado estándar (media=0, varianza=1)
# Esto es importante porque muchos modelos (especialmente los basados en distancia y redes neuronales) requieren que los datos estén normalizados
numerical_transformer = StandardScaler()

# Creamos el transformador para variables categóricas: codificación one-hot
# handle_unknown='ignore' permite que categorías no vistas en entrenamiento no generen error en predicción
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combinamos ambos transformadores en un ColumnTransformer
# Esto permite aplicar diferentes transformaciones a columnas específicas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # Aplica escalado a variables numéricas
        ('cat', categorical_transformer, categorical_features)  # Aplica one-hot a categóricas
    ],
    remainder='passthrough'  # Deja pasar columnas no especificadas (por si acaso)
)

print("Pipeline de preprocesamiento definido exitosamente.\n")

# -----------------------------
# 5. Configuración de validación cruzada y función de evaluación
# -----------------------------

# Usamos StratifiedKFold para mantener la proporción de clases en cada fold
# n_splits=5 es estándar y da buen equilibrio entre sesgo y varianza
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definimos una función para evaluar modelos usando validación cruzada
# Calcula métricas relevantes para clasificación desbalanceada: F1, AUC-ROC, precisión, recall, accuracy
# Devuelve promedios y desviaciones estándar para cada métrica

def evaluate_model(model, X_data, y_data, model_name="Modelo"):
    f1_scores, auc_roc_scores = [], []
    accuracy_scores, precision_scores, recall_scores = [], [], []

    print(f"\n--- Evaluando {model_name} ---")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_data, y_data)):
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]

        # Entrenamos el pipeline completo (incluye preprocesamiento y SMOTE solo en entrenamiento)
        model.fit(X_train, y_train)

        # Realizamos predicciones sobre el conjunto de validación
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]  # Probabilidad de la clase positiva

        # Calculamos métricas
        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        auc_roc_scores.append(roc_auc_score(y_val, y_proba))

        print(f"  Fold {fold+1}: F1 = {f1_scores[-1]:.3f}, AUC = {auc_roc_scores[-1]:.3f}")

    print(f"\n{model_name} - Resultados Promedio:")
    print(f"  Accuracy: {np.mean(accuracy_scores):.3f} +/- {np.std(accuracy_scores)*2:.3f} (95% CI)")
    print(f"  Precisión: {np.mean(precision_scores):.3f} +/- {np.std(precision_scores)*2:.3f} (95% CI)")
    print(f"  Recall: {np.mean(recall_scores):.3f} +/- {np.std(recall_scores)*2:.3f} (95% CI)")
    print(f"  F1-Score: {np.mean(f1_scores):.3f} +/- {np.std(f1_scores)*2:.3f} (95% CI)")
    print(f"  AUC-ROC: {np.mean(auc_roc_scores):.3f} +/- {np.std(auc_roc_scores)*2:.3f} (95% CI)")

    return {
        'F1-Score': np.mean(f1_scores),
        'AUC-ROC': np.mean(auc_roc_scores),
        'F1-CI': np.std(f1_scores) * 2,
        'AUC-CI': np.std(auc_roc_scores) * 2
    }

# -----------------------------
# 6. Implementación y entrenamiento de modelos
# -----------------------------

# Todos los modelos se implementan usando ImbPipeline que incluye:
# 1. Preprocesamiento (escalado numérico, codificación one-hot categórico)
# 2. SMOTE (aplicado solo a datos de entrenamiento en cada fold)
# 3. El clasificador específico

print("\n--- Iniciando entrenamiento y evaluación de modelos ---")

# -----------------------------
# Modelo 1: Regresión Logística
# -----------------------------
from sklearn.linear_model import LogisticRegression

print("\n--- Modelo 1: Regresión Logística ---")

# Clase auxiliar para convertir matrices sparse a densas (necesario para algunos modelos)
class DenseTransformer():
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray()

# Pipeline para Regresión Logística
pipeline_lr = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento (escalado + one-hot)
    ('smote', SMOTE(random_state=42)),  # Balancea clases usando sobremuestreo sintético
    ('classifier', LogisticRegression(
        random_state=42,  # Para reproducibilidad
        solver='liblinear',  # Optimizador eficiente para datasets pequeños/medianos
        penalty='l1',  # Regularización L1 (Lasso) para selección de características
        max_iter=1000  # Máximo número de iteraciones para convergencia
    ))
])

# Grid de hiperparámetros para búsqueda
# C: parámetro de regularización (inverso de lambda)
# Valores más pequeños = más regularización, valores más grandes = menos regularización
param_grid_lr = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]  # Rango logarítmico para explorar diferentes niveles de regularización
}

# Búsqueda de hiperparámetros usando validación cruzada
grid_search_lr = GridSearchCV(
    pipeline_lr, 
    param_grid_lr, 
    cv=cv,  # Usa la validación cruzada definida anteriormente
    scoring='f1',  # Optimiza F1-score (importante para clases desbalanceadas)
    n_jobs=-1,  # Usa todos los núcleos disponibles
    verbose=1  # Muestra progreso
)
grid_search_lr.fit(X, y)

print(f"Mejores parámetros para Regresión Logística: {grid_search_lr.best_params_}")

# Obtiene el mejor modelo y lo evalúa
best_lr_model = grid_search_lr.best_estimator_
results_lr = evaluate_model(best_lr_model, X, y, "Regresión Logística Optimizada")

# -----------------------------
# Modelo 2: K-Nearest Neighbors (KNN)
# -----------------------------
from sklearn.neighbors import KNeighborsClassifier

print("\n--- Modelo 2: K-Vecinos Más Cercanos (KNN) ---")

# Pipeline para KNN
pipeline_knn = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento
    ('smote', SMOTE(random_state=42)),  # Balancea clases
    ('classifier', KNeighborsClassifier())  # Clasificador KNN
])

# Grid de hiperparámetros para KNN
param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11],  # Número de vecinos (valores impares para evitar empates)
    'classifier__weights': ['uniform', 'distance'],  # Pesos: uniforme o por distancia
    'classifier__p': [1, 2]  # Distancia: 1=Manhattan, 2=Euclidiana
}

# Búsqueda de hiperparámetros
grid_search_knn = GridSearchCV(
    pipeline_knn, 
    param_grid_knn, 
    cv=cv, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_search_knn.fit(X, y)

print(f"Mejores parámetros para KNN: {grid_search_knn.best_params_}")
best_knn_model = grid_search_knn.best_estimator_
results_knn = evaluate_model(best_knn_model, X, y, "KNN Optimizado")

# -----------------------------
# Modelo 3: Random Forest (Bosque Aleatorio)
# -----------------------------
from sklearn.ensemble import RandomForestClassifier

print("\n--- Modelo 3: Random Forest ---")

# Pipeline para Random Forest
pipeline_rf = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento
    ('smote', SMOTE(random_state=42)),  # Balancea clases
    ('classifier', RandomForestClassifier(random_state=42))  # Clasificador Random Forest
])

# Grid de hiperparámetros para Random Forest
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],  # Número de árboles en el bosque
    'classifier__max_depth': [10, 20, 30, None],  # Profundidad máxima de cada árbol
    'classifier__min_samples_leaf': [1, 2, 4],  # Mínimo número de muestras en hojas
    'classifier__max_features': ['sqrt', 'log2', 0.5],  # Número de características a considerar en cada split
}

# Búsqueda de hiperparámetros
grid_search_rf = GridSearchCV(
    pipeline_rf, 
    param_grid_rf, 
    cv=cv, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_search_rf.fit(X, y)

print(f"Mejores parámetros para Random Forest: {grid_search_rf.best_params_}")
best_rf_model = grid_search_rf.best_estimator_
results_rf = evaluate_model(best_rf_model, X, y, "Random Forest Optimizado")

# -----------------------------
# Modelo 4: Red Neuronal Artificial (MLP)
# -----------------------------
from sklearn.neural_network import MLPClassifier

print("\n--- Modelo 4: Red Neuronal Artificial (MLP) ---")

# Pipeline para MLP
pipeline_mlp = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento
    ('smote', SMOTE(random_state=42)),  # Balancea clases
    ('classifier', MLPClassifier(
        random_state=42,  # Para reproducibilidad
        max_iter=1000,  # Máximo número de iteraciones
        early_stopping=True,  # Detiene temprano si no hay mejora
        validation_fraction=0.1,  # Usa 10% para validación temprana
        n_iter_no_change=10,  # Detiene si no hay mejora por 10 iteraciones
        tol=1e-4  # Tolerancia para convergencia
    ))
])

# Grid de hiperparámetros para MLP
param_grid_mlp = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],  # Arquitectura de la red
    'classifier__activation': ['relu'],  # Función de activación
    'classifier__solver': ['adam'],  # Optimizador
    'classifier__alpha': [0.0001, 0.001, 0.01],  # Regularización L2
    'classifier__learning_rate': ['constant']  # Tasa de aprendizaje
}

# Búsqueda de hiperparámetros
grid_search_mlp = GridSearchCV(
    pipeline_mlp, 
    param_grid_mlp, 
    cv=cv, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_search_mlp.fit(X, y)

print(f"Mejores parámetros para MLP: {grid_search_mlp.best_params_}")
best_mlp_model = grid_search_mlp.best_estimator_
results_mlp = evaluate_model(best_mlp_model, X, y, "MLP Optimizado")

# -----------------------------
# Modelo 5: Support Vector Machine (SVM)
# -----------------------------
from sklearn.svm import SVC

print("\n--- Modelo 5: Support Vector Machine (SVM) ---")

# Pipeline para SVM
pipeline_svc = ImbPipeline(steps=[
    ('preprocessor', preprocessor),  # Aplica preprocesamiento
    ('smote', SMOTE(random_state=42)),  # Balancea clases
    ('classifier', SVC(
        random_state=42,  # Para reproducibilidad
        probability=True  # Necesario para obtener probabilidades
    ))
])

# Grid de hiperparámetros para SVM
param_grid_svc = {
    'classifier__C': [0.1, 1, 10],  # Parámetro de regularización
    'classifier__kernel': ['linear', 'rbf', 'poly'],  # Tipo de kernel
    'classifier__gamma': ['scale', 'auto', 0.1, 1]  # Coeficiente del kernel
}

# Búsqueda de hiperparámetros
grid_search_svc = GridSearchCV(
    pipeline_svc, 
    param_grid_svc, 
    cv=cv, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_search_svc.fit(X, y)

print(f"Mejores parámetros para SVM: {grid_search_svc.best_params_}")
best_svc_model = grid_search_svc.best_estimator_
results_svc = evaluate_model(best_svc_model, X, y, "SVM Optimizado")

# -----------------------------
# 7. Resumen de resultados
# -----------------------------

print("\n--- Todas las evaluaciones de modelos completadas ---")

# Recopilamos todos los resultados para generar tabla de resumen
all_results = {
    "Regresión Logística": results_lr,
    "KNN": results_knn,
    "Random Forest": results_rf,
    "MLP": results_mlp,
    "SVM": results_svc
}

print("\n--- Resumen del rendimiento de los mejores modelos ---")
for model_name, metrics in all_results.items():
    print(f"{model_name}: F1-Score = {metrics['F1-Score']:.3f} +/- {metrics['F1-CI']:.3f}, AUC-ROC = {metrics['AUC-ROC']:.3f} +/- {metrics['AUC-CI']:.3f}")

# -----------------------------
# 8. Análisis de resultados y conclusiones
# -----------------------------

print("\n--- Análisis de resultados ---")

# Encontramos el mejor modelo basado en F1-Score
best_model_name = max(all_results.keys(), key=lambda x: float(all_results[x]['F1-Score']))
best_f1_score = all_results[best_model_name]['F1-Score']

print(f"El mejor modelo según F1-Score es: {best_model_name} con F1 = {best_f1_score:.3f}")

# Encontramos el mejor modelo basado en AUC-ROC
best_auc_model = max(all_results.keys(), key=lambda x: float(all_results[x]['AUC-ROC']))
best_auc_score = all_results[best_auc_model]['AUC-ROC']

print(f"El mejor modelo según AUC-ROC es: {best_auc_model} con AUC = {best_auc_score:.3f}")

print("\n--- Fin del análisis de modelos ---")

