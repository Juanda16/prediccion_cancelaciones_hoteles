# Predicción de cancelaciones en reservaciones de hoteles

**Este proyecto utiliza técnicas de _machine learning_ para predecir cancelaciones en reservaciones de hoteles.**  
El objetivo es ayudar a los hoteles a anticipar cancelaciones y optimizar la gestión de sus reservas.

---

## 📑 **Tabla de Contenidos**
- [Descripción](#descripción)
- [Instalación](#instalación)
- [Uso](#uso)
- [Datos](#datos)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## Descripción 📝
La predicción de cancelaciones es crucial para la industria hotelera.  
Este proyecto analiza datos históricos de reservaciones y desarrolla una **solución integral de machine learning** para predecir cancelaciones de reservas hoteleras, implementando y comparando múltiples algoritmos con técnicas avanzadas de selección y extracción de características.

### 🎯 Objetivos Principales

- **Evaluar** 5 algoritmos de machine learning para predicción de cancelaciones
- **Implementar** técnicas de selección secuencial de características
- **Aplicar** extracción de características mediante PCA
- **Comparar** rendimiento con el estado del arte
- **Desarrollar** una solución lista para implementación



### 🎯 Mejor Modelo: **Random Forest**
- **F1-Score**: 0.814 ± 0.014
- **AUC-ROC**: 0.933 ± 0.014
- **Precisión**: 0.838 ± 0.019
- **Recall**: 0.791 ± 0.016

---

## Instalación ⚙️ 

1. **Clona este repositorio:**  
   git clone https://github.com/Juanda16/prediccion_cancelaciones_hoteles.git

2. **Instala las dependencias necesarias:**  
   pip install -r requirements.txt o pip3 install -r requirements.txt

---

### 3. **Análisis Específicos**

```bash
# Selección secuencial de características
python/python3 scripts/seleccion_caracteristicas/seleccion_secuencial_robusta.py

# Extracción PCA
python/python3 scripts/extraccion_pca/extraccion_caracteristicas_pca_final.py


## Uso 🚀 

Ejecuta el script principal para entrenar el modelo y hacer predicciones:  
python main.py

_Asegúrate de tener el archivo de datos en la carpeta correspondiente._

---

## Datos 📊 

- **Fuente:** [Incluye aquí la fuente del dataset si es posible]
- **Preprocesamiento:** Limpieza de datos, manejo de valores nulos y transformación de variables categóricas.

---

## Metodología 🧠 

- **Exploración de datos:** Análisis estadístico y visualización.
- **Modelos utilizados:** Regresión logística, Random Forest, XGBoost.
- **Evaluación:** Precisión, recall, matriz de confusión, ROC-AUC.

---

## Resultados 📈 

### 📊 Resultados Principales

| Modelo | F1-Score | AUC-ROC | Accuracy | Recomendación |
|--------|----------|---------|----------|---------------|
| **Random Forest** | **0.814** | **0.933** | 0.863 | ✅ **Producción** |
| **SVM** | **0.797** | **0.923** | 0.846 | ✅ **Alternativa** |
| Logistic Regression | 0.768 | 0.900 | 0.825 | ⚠️ Baseline |
| KNN | 0.744 | 0.880 | 0.814 | ❌ No recomendado |
| MLP | 0.733 | 0.892 | 0.813 | ❌ No recomendado |


---

## Contribuciones 🤝 

¡Las contribuciones son bienvenidas!  
Por favor, abre un _issue_ o un _pull request_ para proponer mejoras o correcciones.

---

## Licencia 📝 

Este proyecto está bajo la licencia **MIT**. Consulta el archivo LICENSE para más información.

---
