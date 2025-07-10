# Predicción de cancelaciones en reservaciones de hoteles

**Este proyecto utiliza técnicas de _machine learning_ para predecir cancelaciones en reservaciones de hoteles.**  
El objetivo es ayudar a los hoteles a anticipar cancelaciones y optimizar la gestión de sus reservas.

---

## 📑 **Tabla de Contenidos**
- [Descripción](#descripción-📝)
- [VIdeo Explicativo](#link-al-video-explicativo-video-de-presentación)
- [Instalación](#instalación)
- [Análisis Específicos](#análisis-específicos)
- [Uso](#uso-🚀)
- [Datos](#datos-📊)
- [Metodología](#metodología-🧠)
- [Resultados](#resultados-📈)
- [Contribuciones](#contribuciones-🤝)
- [Licencia](#licencia-📝)

---

## Descripción 📝
La predicción de cancelaciones es crucial para la industria hotelera.  
Este proyecto analiza datos históricos de reservaciones y desarrolla una **solución integral de machine learning** para predecir cancelaciones de reservas hoteleras, implementando y comparando múltiples algoritmos con técnicas avanzadas de selección y extracción de características.

## Link al video explicativo: [Video de presentación](https://drive.google.com/file/d/1GXIzq5xeMyxDU0jvkdIBEiI1gAH-m4Ng/view?usp=sharing)

### 🎯 Objetivos Principales

- **Evaluar** 5 algoritmos de machine learning para predicción de cancelaciones
- **Implementar** técnicas de selección secuencial de características
- **Aplicar** extracción de características mediante PCA
- **Comparar** rendimiento con el estado del arte
- **Desarrollar** una solución lista para implementación

---

## Instalación

1. **Clona este repositorio:**  
   git clone https://github.com/Juanda16/prediccion_cancelaciones_hoteles.git

2. **Instala las dependencias necesarias:**  
   pip install -r requirements.txt o pip3 install -r requirements.txt

---
## Uso 🚀 

Ejecuta el script principal para entrenar el modelo y hacer predicciones:  
python analisis_modelos.py

_Asegúrate de tener el archivo de datos en la carpeta correspondiente._

---

### Análisis Específicos

```bash
# Selección secuencial de características
python/python3 scripts/seleccion_caracteristicas/seleccion_secuencial_robusta.py

# Extracción PCA
python/python3 scripts/extraccion_pca/extraccion_caracteristicas_pca_final.py
```

---



## Datos 📊 

- **Fuente:** [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Preprocesamiento:** 
  - Limpieza de datos
  - Manejo de valores faltantes
  - Codificación de variables categóricas
  - Normalización de variables numéricas

---

## Metodología 🧠 

- **Exploración de datos:** Análisis estadístico y visualización.
- **Modelos utilizados:** Regresión logística, KNN, Random Forest, MLP, SVM.
- **Selección de características:**
  - Selección secuencial de características para reducir dimensionalidad.
- **Extracción de características:**
  - PCA para identificar componentes principales y reducir ruido.
- **Entrenamiento y validación:**
  - División de datos en conjuntos de entrenamiento y prueba.
- **Hiperparámetros:** Búsqueda de hiperparámetros óptimos mediante validación cruzada.
- **Evaluación de modelos:**
  - Métricas de rendimiento: F1-Score, AUC-ROC, precisión, recall, accuracy.
  - Comparación de modelos para seleccionar el mejor.


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