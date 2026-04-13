# Análisis de Interrupciones Aéreas ✈️

![Python Minimum Version](https://img.shields.io/badge/Python->=3.8-blue?logo=python&logoColor=white)
![Streamlit App](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&logoColor=white)
![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)

> Este proyecto fue creado por **Mariana Patiño Múnera**.  
> Puedes encontrar más proyectos y repositorios visitando mi perfil en GitHub: [https://github.com/marianapatinom](https://github.com/marianapatinom).

## Descripción del Proyecto

Este repositorio contiene los resultados del análisis y las visualizaciones implementadas respecto al **Airline Disruptions Dataset**. El flujo de trabajo del proyecto abarcó un proceso de ETL completo en un *Jupyter Notebook* y ahora se materializa a través de un panel de inteligencia de datos (Dashboard) profesional desarrollado íntegramente en Python con Streamlit.

## Componentes

1. **`Parcial1_Mariana_Patiño.ipynb`**: Cuaderno original de Kaggle con los procesos de: Extracción de datos, depuración, corrección de tipos, variables derivadas, visualización básica con Plotly (Estadística Descriptiva y Boxplots) y hallazgos clave.
2. **`index.html`**: Una vistosa *"Landing Page"* diseñada para ilustrar al usuario foráneo los pasos ejecutados durante el proceso de Ciencia de Datos. Es responsiva y de corte profesional.
3. **`app.py`**: Aplicación de Streamlit equipada con navegación en fichas (tabs), que ofrece un compendio interactivo a través de filtros en una barra lateral (Sidebar). Soporta gráficos estadísticos de alta densidad, un mapa interactivo global y un motor de predicción dinámico con Machine Learning. 

## Instalación y Ejecución Local

Sigue estos pasos para correr la aplicación Streamlit en tu máquina local:

1. **Clona el repositorio** en un directorio de tu computadora.
   ```bash
   git clone https://github.com/marianapatinom/... # Asegurate de completar el nombre de la URL exacta de este repositorio.
   ```
2. **Instala las dependencias** recomendadas en tu entorno virtual.
   ```bash
   pip install -r requirements.txt
   ```
3. **Ejecuta la aplicación Streamlit**
   ```bash
   streamlit run app.py
   ```

Una vez que el comando inicie, Streamlit te dará un enlace local (generalmente `http://localhost:8501`) donde podrás interactuar libre y dinámicamente con los módulos visuales y el análisis predictivo del impacto operativo en aerolíneas!

## Estructura de Fichas (app.py)
* **🗂️ Datos y Frecuencias**: Tabla del dataset renderizada, unida a frecuencias comparativas de aerolíneas vs países.
* **📈 Análisis Exploratorio**: Visualizaciones profundas mediante Histogramas y Boxplots creados a partir de `plotly.express`.
* **🌍 Mapa de Impacto**: Despliegue global en 2D mapeando las ubicaciones de los siniestros y redireccionamientos.
* **🤖 Análisis Predictivo**: Permite alimentar métricas hipotéticas o reales en cajas configurables para calcular instantáneamente el impacto financiero estimado (Estimación Predictiva usando *scikit-learn*).

---
**Realizado por Mariana Patiño Múnera**

