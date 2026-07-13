# Índice Temático de Notebooks (Semanas 9-15)

Este archivo sirve como guía de navegación interactiva para todos los temas, modelos y ejercicios prácticos correspondientes a la segunda mitad del curso. 

---

## 🛡️ SVM (Support Vector Machines) y SMOTE
*   **[12_ML_Clasificador_SVM_Falla_Cardiaca.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_9/12_ML_Clasificador_SVM_Falla_Cardiaca.ipynb):** Notebook principal de SVM. Contiene la explicación matemática del hiperplano marginal, ajuste de parámetros `C` y `gamma` con `GridSearchCV`, y el uso de **SMOTE** para balancear clases en el set de Train.
*   **[MachineLearningSEM9_SVM.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_9/MachineLearningSEM9_SVM.ipynb):** Ejemplo complementario básico de entrenamiento de SVM.

---

## 👥 Ensambles (Bagging, Boosting y Voting)
*   **[14_ML_Ensamble_Clasificadores.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_10/14-Ejercicio-Ensambles-Deteccion-Diebetes/14_ML_Ensamble_Clasificadores.ipynb):** Ensamble de clasificadores. Contiene gráficos de correlación, distribución, histograma, reducción de dimensiones (UMAP), y la comparación teórica de `VotingClassifier` (voto blando), `BaggingClassifier` y `AdaBoostClassifier`.
*   **[13_ML_Ensembles_Random_Forests.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_10/13-Ejercicio-Ensembles-RandomForests/13_ML_Ensembles_Random_Forests.ipynb):** Entrenamiento de Random Forest sin optimizar parámetros y utilizando validación cruzada y parámetros optimizados (`GridSearchCV`). Incluye la resolución de la tarea final sobre deciles y mapeo de scores para estrategias bancarias.
*   **[esemble.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_10/esemble.ipynb):** Celda complementaria simple de ensambles.

---

## 📈 Regresiones (Lineal y Logística)
*   **[Sem11_RegresionLineal.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_11/Sem11_RegresionLineal.ipynb):** Regresión lineal simple para predecir variables continuas ($y = mx + b$). Explica el cálculo de pendiente, intercepto, MSE y $R^2$.
*   **[Sem11_RegresionLogistica.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_11/Sem11_RegresionLogistica.ipynb):** Regresión logística binaria. Explica la función sigmoide, el mapeo a probabilidad de 0 a 1, y el umbral de decisión para clasificar.
*   **[RegresionLineal.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_11/RegresionLineal.ipynb):** Archivo complementario de regresión lineal.

---

## 🏷️ Clustering (K-Means y Jerárquico)
*   **[16_01_ML_Segmentacion_de_Clientes.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_12/16_01_ML_Segmentacion_de_Clientes.ipynb):** Notebook completo de segmentación de clientes con K-Means. Contiene un gran análisis exploratorio de los datos (EDA), escalado StandardScaler, reducción PCA y mapeo de grupos.
*   **[Ejemplo 1_ Segmentación de Clientes (K-Means)_Analisis_Codo-Silueta.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_12/Ejemplo%201_%20Segmentaci%C3%B3n%20de%20Clientes%20(K-Means)_Analisis_Codo-Silueta.ipynb):** Evaluación de K-Means con y sin el Método del Codo (Inercia/WCSS) y el Coeficiente de Silueta.
*   **[Ejemplo 2_ Agrupamiento de Estaciones Meteorológicas (Jerárquico).ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_12/Ejemplo%202_%20Agrupamiento%20de%20Estaciones%20Meteorol%C3%B3gicas%20(Jer%C3%A1rquico).ipynb):** Notebook de Agrupamiento Jerárquico. Contiene la simulación meteorológica, escalado, comparación de dendrogramas Ward vs. Complete, correlación cofenética, y entrenamiento de `AgglomerativeClustering`.

---

## 🛒 Reglas de Asociación
*   **[17_01_ML_Reglas_de_Asociacion_APriori.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_13/17_01_ML_Reglas_de_Asociacion_APriori.ipynb):** Algoritmo Apriori para reglas de asociación de canastas de compra. Explica las métricas de Soporte, Confianza y Lift.
*   **[Sem13_Ejm1_AlgoritmoApriori.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_13/Sem13_Ejm1_AlgoritmoApriori.ipynb):** Ejemplo básico de Apriori.
*   **[Sem13_Ejm2_Algoritmo-Apriori_Recomendación de Contenido.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_13/Sem13_Ejm2_Algoritmo-Apriori_Recomendaci%C3%B3n%20de%20Contenido.ipynb):** Ejemplo de Apriori adaptado a la recomendación de películas o contenidos.

---

## 🤝 Aprendizaje Semi-Supervisado
*   **[17_02_aprendizaje_semisupervisado.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_14/17_02_aprendizaje_semisupervisado.ipynb):** Enfoque semi-supervisado usando K-Means para encontrar imágenes representativas, etiquetado manual y propagación de etiquetas (*Label Propagation*) a todo el dataset.
*   **[17_03_aprendizaje_semisupervisado_fashion_mnist.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_14/17_03_aprendizaje_semisupervisado_fashion_mnist.ipynb):** Ejemplo similar de propagación de etiquetas aplicado al dataset de imágenes Fashion MNIST.
*   **[Sem13_SelfTraining_ClasificacionDocumentos.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_14/Sem13_SelfTraining_ClasificacionDocumentos.ipynb):** Clasificación semi-supervisada usando el algoritmo `SelfTrainingClassifier` en texto.
*   **[Sem13_DiagnosticoMedico.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_14/Sem13_DiagnosticoMedico.ipynb):** Diagnóstico semi-supervisado aplicado al ámbito médico.

---

## 🕹️ Aprendizaje por Refuerzo (Reinforcement Learning)
*   **[18_01_aprendizaje_por_refuerzo_Q_Learning.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_15/18_01_aprendizaje_por_refuerzo_Q_Learning.ipynb):** Método Q-Learning clásico usando el entorno `FrozenLake` de Gym. Explica la Q-Table, la Ecuación de Bellman, y el balance exploración-explotación ($\epsilon$-greedy).
*   **[RL_Agente-Entorno-Modelo_V3.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_15/RL_Agente-Entorno-Modelo_V3.ipynb):** Implementación de la interacción Agente-Entorno.
*   **[RL_Agente-Entorno_V1.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_15/RL_Agente-Entorno_V1.ipynb):** Versión básica 1 de interacción.
*   **[RL_Agente-Entorno_V2.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Sem_15/RL_Agente-Entorno_V2.ipynb):** Versión básica 2 de interacción.

---

## 🎓 Casos Especiales de Simulación de Examen Final
*   **[Examen_Final_Caso1_Supervisado_Completo.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Examen_Final_Caso1_Supervisado_Completo.ipynb):** Simulación de examen final enfocada en **Aprendizaje Supervisado** (KNN, SVM, Naive Bayes, Random Forest, AdaBoost, MLP, Regresión Logística y Regresión Lineal).
*   **[Examen_Final_Caso2_NoSupervisado_Completo.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Examen_Final_Caso2_NoSupervisado_Completo.ipynb):** Simulación de examen final enfocada en **Aprendizaje No Supervisado** (K-Means, Hierarchical Clustering, PCA y SOM).
*   **[Examen_Final_Caso3_SemiSupervisado.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Examen_Final_Caso3_SemiSupervisado.ipynb):** Simulación de examen final enfocada en **Aprendizaje Semi-Supervisado** (Label Propagation y Self-Training).
*   **[Examen_Final_Caso4_Refuerzo.ipynb](file:///c:/Users/Daniel/Documents/UPC/machine%20learning/examen/ML/Examen_Final_Caso4_Refuerzo.ipynb):** Simulación de examen final enfocada en **Aprendizaje por Refuerzo** (Q-Learning y optimización por grilla de hiperparámetros).
