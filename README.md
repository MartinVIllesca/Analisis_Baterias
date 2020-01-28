# Analisis_Baterias

En este repositorio se encuentran los códigos utilizados en el análisis de datos de una batería descargada con caminatas aleatoria de corriente constante. La base de [datos](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) (correspondiente a la número 2 del dataset 11). También se encuentra el informe donde se explica en detalle el trabajo realizado.

Para utilizar el código, se debe descargar el repositorio y los datos. Se debe descomprimir el set de datos en la misma carpeta donde se descomprime el repositorio. Además, de instalar las dependencias de python para utilizar el código; estas son:

- python 3.6
- matplotlib
- numpy
- pandas
- sklearn
- scipy

Además de instalar `jupyter` para correr los notebooks.

El código se encuentra organizado en carpetas de notebooks y Scripts. En la primera se encuentran los notebooks donde
se realiza el análisis de los datos y la creación de los modelos; dentro de la carpeta entrega se encuentran los notebooks
donde el código está actualizado según lo realizado en el informe. En la carpeta de Scripts, se encuentran scripts
de procesamiento de datos auxiliares para la extracción de las características y la carga desde los archivos de datos.

Para revisar los códigos, se recomienda comenzar por la `visualizaciones` de los datos y las características extraídas, para seguir por el de `Trabajo Dirigido`. El notebook de `Prueba SBM` se trata de la exploración del modelo SBM, y finalmente, el notebook de `exploración_Cesar` trata de los resultados finales obtenidos con el clasificador/regresor.

Mails de contacto: Martín Valderrama valderramaillesca.m@gmail.com, César Baeza cesarbaeza1995@gmail.com
