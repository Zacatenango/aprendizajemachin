# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:40:08 2023

@author: zaratejo
"""

# -*- coding: utf-8 -*-
#%% Import libraries
import os
import pandas as pd
import numpy as np
import sklearn.metrics as sklearn_metrics # similarity metrics
import scipy.spatial.distance as scipy_spatial_distance # distance metrics

#%% Import data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Test de películas (anónimo)(1-12).xlsx')
# data = pd.read_excel(file_path,encoding='latin_1',index_col=0) #old versions
data = pd.read_excel(file_path,index_col=0)
data.head()

#%% Sel7ect columns
# Seleccionamos cada 3 columnas a partir de la columna 9, ya que el acomodo de la tabla es
# <título de la película con las calificaciones que cada usuario puso> - <puntos> - <comentarios>
# Y de ahí sólo nos importan las calificaciones
def select_columns(x):
  csel = np.arange(9,246,3)
  users1 = list(x.iloc[:,6])
  cnames1 = list(x.columns.values[csel])
  x = x[cnames1]
  x.index = users1
  
  return x

data_seleccionada =  select_columns(data)
data_seleccionada.head()
data_seleccionada_original = data_seleccionada.copy()


#%% Average rating of the movies
# Sacamos unas estadísticas básicas: el promedio de calificación de cada película, y el de cada 
# persona (nos interesa saber si alguien es pesimista)
movie_prom = data_seleccionada.mean(axis=0)
user_prom = data_seleccionada.mean(axis=1)

#%% Change the stars to like or dislike
# Convertimos los datos a binario: tomamos como like una calificación de 4 o más
nombres_de_columnas = list(data_seleccionada.columns.values)
fnames = np.array(data_seleccionada.index)
for col in nombres_de_columnas:
    data_seleccionada[col]=np.where(data_seleccionada[col]>3,1,0)
data_seleccionada.head()
datan_binarizado = data_seleccionada.copy()

#%% Calculate similarity indices in users by sklearn
# Usamos para eso los likes de datan_binarizado. Tomamos <datan_binarizado.iloc[0,:]>, que son las
# calificaciones del usuario 0 (Mary11), como el conjunto "verdadero"; y como el conjunto "predicho"
# tomamos <datan_binarizado.iloc[1,:]>, que son las calificaciones de Walky (usuario 1).
# Lo hacemos para determinar la similitud entre ambos, En los siguientes renglones sacaremos 2
# diferentes índices de similitud.
cf_m = sklearn_metrics.confusion_matrix(datan_binarizado.iloc[0,:],datan_binarizado.iloc[1,:])

# Aquí usamos la similitud simple
# Sean:
# - a = registros falsos en A y B
# - b = registros falsos en A y ciertos en B
# - c = registros ciertos en A y falsos en B
# - d = registros ciertos en A y B
# La similitud simple es (a + b) / (a+b+c+d) (aciertos / total)
sim_simple = sklearn_metrics.accuracy_score(data_seleccionada.iloc[0,:],data_seleccionada.iloc[1,:])
#sim_simple_new = (cf_m[0,0]+cf_m[1,1])/np.sum(cf_m)
print('Simple : %0.4f'%sim_simple)

# Índice de Jaccard: d / (b+c+d) - películas que gustan a ambos / total excluyendo las que no gustan
# Esto limita la similitud a únicamente casos positivos; 
sim_jac = sklearn_metrics.jaccard_score(data_seleccionada.iloc[0,:],data_seleccionada.iloc[1,:])
sim_jac = (cf_m[0,0])/(np.sum(cf_m)-cf_m[1,1])
print('Jaccard: %0.4f'%sim_jac)

cf_m_binaria = cf_m

# Tip for those who have a different syntax
# conda update sklearn

#%% Calculation of distances by scipy
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
# Sacamos 2 tipos de distancia: el teorema de Pitágoras (euclidiana), y la distancia Canberra
d1 = scipy_spatial_distance.euclidean(data_seleccionada.iloc[0,:],data_seleccionada.iloc[5,:])
print('Simple : %0.4f'%d1)
# La distancia Canberra es una variante ponderada de la distancia Manhattan. Sean p = (p1, ..., pn)
# y q = (q1, q2, ..., qn) 2 vectores en R^n; su distancia Canberra es:
#    d(p,q) = sum for i = 1 to n of [ abs(p[i] - q[i]) / ( abs(p[i]) + abs(q[i]) ) ]
# Dividir entre la suma de las dimensiones hace que la distancia calculada varíe mucho cuando las
# dimensiones son muy pequeñas y cercanas al 0 (ya que dividir entre algo muy pequeño da algo muy
# grande), lo que permite detectar cambios minúsculos en dimensiones pequeñas.
d2 = scipy_spatial_distance.canberra(data_seleccionada.iloc[0,:],data_seleccionada.iloc[5,:])
print('Canberra: %0.4f'%d2)

#%% Calculate all possible combinations by scipy
# Sacamos aquí la distancia entre pares de elementos
# scipy.spatial.distance.pdist() recibe un arreglo de M x N que representa M observaciones de N
# dimensiones, y tira una matriz de distancias. Cada fila representa una observación, y cada columna
# representa la distancia de nuestra observación a las demás. La matriz es cuadrada de M x M porque
# ésta representa la distancia de M observaciones hacia las demás M observaciones. La diagonal
# siempre es 0, porque cada item de la diagonal representa la distancia de un elemento a sí mismo.
# Para este cálculo de distancia vamos a usar la distancia Hamming, que es la cantidad de valores
# diferentes entre cada elemento; es óptima para datos binarios, como data_seleccionada binarizada.
# Fuente: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
D1 = scipy_spatial_distance.pdist(data_seleccionada,'matching')
# pdist() no tira por sí mismo la matriz de distancia, sino que tira un vector de M^2 observaciones
# Entonces, para tener la matriz de distancia en forma cuadrada, necesitamos squareform(). No es
# necesario proveer el tamaño, porque sabemos que las matrices de distancia siempre son cuadradas
# y por lo tanto la dimensión es la raíz cuadrada de la longitud del vector de observaciones.
D1 = scipy_spatial_distance.squareform(D1)

# Idem, pero con distancia de Jaccard
D2 = scipy_spatial_distance.pdist(data_seleccionada,'jaccard')
D2 = scipy_spatial_distance.squareform(D2)

#%% Select a user and determine the other most similar user
# Con esa matriz de distancias, sacamos un usuario y sacamos las distancias respecto a los demás
# usuarios. Sacamos también con numpy.argsort() los índices de las distancias de los demás respecto
# al usuario <user> en orden ascendente.
user = 1
D_user = D1[user]
D_user_sort = np.sort(D_user)
indx_user = np.argsort(D_user)


#%% Recommendation version 1. The most similar user
# Sacamos un usuario. Primero sacamos su username de fnames (lo sacamos al principio del código), 
# eso lo metemos al pseudoarreglo DataFrame.loc[] que sirve para obtener items a través de su 
# nombre. Con ese mismo método, sacamos de indx_user[] el índice del usuario más similar, que es el 
# #1 porque el #0 siempre va a ser sí mismo. Convertimos eso a nombre de usuario con fnames[], y de 
# ahí sacamos sus datos con data_seleccionada.loc[].
# Tanto User como User_similar son DFs de una columna, y cada fila tiene el nombre de una película 
# como índice y si al usuario le gustó como valor.
User = data_seleccionada.loc[fnames[user]]
User_similar = data_seleccionada.loc[fnames[indx_user[1]]]
User_version1 = User.copy()

# Convertimos las preferencias de User_similar a True/False donde True significa "1" (like), y 
# las de User a True/False donde True significa 0 (no like). Esto nos da las películas que User
# no le ha puesto al menos 4 estrellas (lo que incluye las que no ha visto) y que User_similar sí
# le ha dado al menos 4 en forma de un DataFrame booleano con índices.
# Y con eso acabo de sacar las películas qué recomendar a User.
indx_recomen = (User_similar ==1)&(User==0)
# Ahora que tengo ese dato, saco los índices de las películas que dieron True y las tiro.
recomend1 = list(User.index[indx_recomen])
print('\n Movie list recommended:\n')
print(recomend1)



#%% Recommendation version 2. The k most similar users
# Ahora sigue la variante 2: sacar los k usuarios más similares. En las redes sociales esto sirve
# para sacar qué posts mostrar. Vamos a sacar k=5.
k = 5
# Comenzamos sacando a nuestro User
User = data_seleccionada.loc[fnames[user]]
User_version2 = User.copy()  # Guardamos una copia del usuario para verla en Data Wrangler
# Tomamos los top K usuarios más similares a nuestro User y sacamos sus películas
Usernames_similares_topK = fnames[indx_user[1:k+1]]
Usernames_similares_topK_peliculas = data_seleccionada.loc[Usernames_similares_topK]
# Agregamos las similitudes a través del promedio
User_similar = np.mean(Usernames_similares_topK_peliculas,axis=0)
User_similar_promedio_top5 = User_similar.copy()
# Los promedios que sean menores o iguales a 0.5 los hacemos 0, los mayores a eso los hacemos 1
User_similar[User_similar<=0.5] = 0
User_similar[User_similar>0.5] = 1

# Terminamos sacando las recomendaciones con el mismo proceso que la vez pasada
indx_recomen = (User_similar ==1)&(User==0)
recomend2 = list(User.index[indx_recomen])
print('\n Movie list recommended:\n')
print(recomend2)


#%% SIMILARITY WITH MULTISTATE VARIABLES
# Nuestros datos seleccionados originalmente tienen un "NA" de PANDAS para denotar ausencia de
# calificación. Para los siguientes cálculos, es necesario rellenar esos NA con un número. Dado que
# nuestras estrellas son de 1 a 5, podemos usar el 0.
data_seleccionada =  select_columns(data)
data_seleccionada.head()
data_seleccionada.fillna(0,inplace=True)
data_seleccionada_narellenado = data_seleccionada.copy()

#%% Multistate similarity metrics
usuario_1 = data_seleccionada.iloc[0,:]
usuario_2 = data_seleccionada.iloc[1,:]

# Ahora nuestra matriz de confusión ya no es 2 x 2, sino 6 x 6, ¿por qué? porque ahora tenemos una
# variable aleatoria X = { 0,1,2,3,4,5 }. Entonces, ahora cada celda [fila, columna] de la matriz
# de confusión significa cuántas películas tuvieron calificación <fila> del usuario_1 y calificación
# <columna> del usuario_2.
cf_m = sklearn_metrics.confusion_matrix(usuario_1, usuario_2)
# Y ahora el accuracy score se saca dividiendo la sumatoria de la diagonal entre la sumatoria de 
# la matriz entera. Es decir, eso me da el porcentaje de películas que fueron calificadas igual por
# ambos usuarios.
sim_simple = sklearn_metrics.accuracy_score(usuario_1, usuario_2)
#sim_simple = sklearn_metrics.accuracy_score(datan.iloc[0,:],datan.iloc[1,:],average='weighted') # old versions
print('Simple : %0.4f'%sim_simple)
# Comparamos eso con el índice de Jaccard
sim_jac = sklearn_metrics.jaccard_score(data_seleccionada.iloc[0,:],data_seleccionada.iloc[1,:],average='weighted')
print('Jaccard : %0.4f'%sim_jac)


#%% GENERATION OF AUXILIARY VARIABLES

# Example of a single variable
# Vamos ahora a generar variables dummy. Tomamos para eso la columna de las calificaciones que tiene
# una de las películas.
calificaciones_de_una_peli = data_seleccionada[nombres_de_columnas[1]]
# Sea X una lista de los resultados de una var aleatoria discreta de valores X = { 1, 2, ..., n }. 
# Una tabla dummy es una tabla cuyas filas son uno de los resultados de la variable, y cuyas
# columnas 1, 2, ..., n indican si el valor de la variable fue 1, 2, ..., n.
dummy1 = pd.get_dummies(calificaciones_de_una_peli)
# dummy1 = pd.get_dummies(datan[cnames[1]],prefix=cnames[1])

#%% Example with users of the entire table
# Aquí vamos a desdoblar las calificaciones que cada usuario le dio a las películas en forma de
# tabla dummy. Comenzamos con un primer saque, las calificaciones de la primera película.
# prefix aquí añade el nombre de la película a la columna. El resultado es una tabla que dice
# con True/False quiénes le dieron 0 a la película 0, quiénes le dieron 1 a la peli 0, &c.
datan_dummy = pd.get_dummies(data_seleccionada[nombres_de_columnas[0]],prefix=nombres_de_columnas[0])
# Hacemos lo mismo con el resto de las películas y pegamos eso al final de las columnas
for col in nombres_de_columnas[1:]:
    tmp = pd.get_dummies(data_seleccionada[col],prefix=col)
    datan_dummy = datan_dummy.join(tmp)
del tmp


#%% DISTANCES WITH QUANTITATIVE VARIABLES
# Ahora probamos sacando la matriz de distancia con diferentes métricas. Volvemos a sacar nuestra
# data_seleccionada para eso
data_seleccionada =  select_columns(data)
data_seleccionada.head()
data_seleccionada.fillna(0,inplace=True)

#%% Euclidean Distance
distancias_pitagoras = scipy_spatial_distance.pdist(data_seleccionada,'euclidean')
distancias_pitagoras = scipy_spatial_distance.squareform(distancias_pitagoras)

#%% Cosine Distance
# Aquí aplicamos el coseno de similitud
# Esto se saca con el coseno del ángulo entre vectores, donde cada vector es una de las filas
distancias_cosenosimilitud = scipy_spatial_distance.pdist(data_seleccionada,'cosine')
distancias_cosenosimilitud = scipy_spatial_distance.squareform(distancias_cosenosimilitud)

#%% Correlation Distance
distancias_correlacion = scipy_spatial_distance.pdist(data_seleccionada,'correlation')
distancias_correlacion = scipy_spatial_distance.squareform(distancias_correlacion)

# %%
# -- Tarea 1: Dado el código anterior, implemente un código que recomiende una película a un usuario
# Pongo esto en función porque hay que hacer esto 3 veces con diferentes métricas de distancia
def recomendar_pelicula(data_seleccionada, fnames, usuario, k, matriz_distancias):
    # Primero sacamos las distancias del usuario objetivo con todos los demás. Las ordenamos, y
    # también sacamos sus índices ordenados
    distancias_usuario = matriz_distancias[usuario]
    distancias_usuario_ordenadas = np.sort(distancias_usuario)
    distancias_usuario_ordenadas_indices = np.argsort(distancias_usuario_ordenadas)

    # Sacamos los 5 usuarios más similares y sus calificaciones
    usuarios_similares_topK = fnames[distancias_usuario_ordenadas_indices[1:k+1]]
    usuarios_similares_topK_peliscalis = data_seleccionada.loc[usuarios_similares_topK]

    # De las películas que mi usuario ha visto, tumbo las que éste haya calificado con 4 o más.
    # Luego tumbo esas películas de las películas con calificaciones de los top 5 más similares
    usuario_peliculas_no_favoritas = data_seleccionada.iloc[usuario][data_seleccionada.iloc[usuario] <= 4]
    usuarios_similares_topK_peliscalis = usuarios_similares_topK_peliscalis.loc[:, usuarios_similares_topK_peliscalis.columns.isin(usuario_peliculas_no_favoritas.index)]

    # Agregamos las calificaciones de los usuarios más similares y ordenamos de menor a mayor
    usuarios_similares_topK_peliscalis_promedio = np.mean(usuarios_similares_topK_peliscalis, axis=0)
    usuarios_similares_topK_peliscalis_promedio_ordenadas = usuarios_similares_topK_peliscalis_promedio.sort_values()

    # Tiramos la película con calificación más alta
    return usuarios_similares_topK_peliscalis_promedio_ordenadas.index[-1]

# Tomamos los k usuarios más similares, y sacamos el promedio de calificaciones. Tomamos a "NATS". 
# Lo hacemos con 3 métricas diferentes
recomendacion_coseno = recomendar_pelicula(data_seleccionada, fnames, 5, 5, distancias_cosenosimilitud)
recomendacion_pitagoras = recomendar_pelicula(data_seleccionada, fnames, 5, 5, distancias_pitagoras)
recomendacion_correlacion = recomendar_pelicula(data_seleccionada, fnames, 5, 5, distancias_correlacion)

# Hecho esto, recomiendo la última película
print(f"Recomendación para el usuario {fnames[5]} (coseno de similitud): {recomendacion_coseno}")
print(f"Recomendación para el usuario {fnames[5]} (teorema de Pitágoras): {recomendacion_pitagoras}")
print(f"Recomendación para el usuario {fnames[5]} (correlación Pearson): {recomendacion_correlacion}")

# Conclusión: aunque nuestras métricas claramente dan resultados distintos, el resultado final es
# el mismo por estar basado en la calificación promedio de los top 5 más similares. Tal vez 
# necesitemos alguna modificación para añadir más variedad.


# %%
# -- Tarea 1 parte 2: Usar los resultados de la encuesta para encontrar el usuario más similar a
# usted y el usuario más diferente, tanto en general como dentro de su propio semestre.
MI_ALIAS = 'Acoyani Garrido Sandoval'

#%% Import de la encuesta
encuesta_path = os.path.join(script_dir, 'Predictive Modeling _ Survey.xlsx')
encuesta = pd.read_excel(encuesta_path)

#%% Selección de las columnas de preguntas Sí/No
# La tabla trae, por cada pregunta, 3 columnas: la pregunta en sí, sus puntos ("Points - ...") y su
# retroalimentación ("Feedback - ..."). De ésas, sólo nos interesan las columnas con la pregunta.
columnas_metadata = ['ID', 'Start time', 'Completion time', 'Email', 'Name', 'Total points',
                      'Quiz feedback', 'Last modified time', 'Alias']
columnas_preguntas = [c for c in encuesta.columns
                       if c not in columnas_metadata
                       and not c.startswith('Points -')
                       and not c.startswith('Feedback -')]

#%% Convertir Sí/No a 1/0, e indexar por Alias
encuesta_binaria = encuesta[columnas_preguntas].apply(lambda col: col.map({'Yes': 1, 'No': 0}))
encuesta_binaria.index = encuesta['Alias']

#%% Determinar en qué semestre respondió cada quien la encuesta
# ITESO tiene 2 semestres al año: Otoño (agosto-diciembre) y Primavera (enero-mayo). Usamos el mes
# de la fecha de inicio de cada respuesta para agrupar a cada persona en su semestre.
def obtener_semestre(fecha):
    periodo = 'Otono' if fecha.month >= 7 else 'Primavera'
    return f'{fecha.year}-{periodo}'

semestre_por_alias = encuesta['Start time'].apply(obtener_semestre)
semestre_por_alias.index = encuesta['Alias']

#%% Ubicar mi respuesta y mi semestre
mi_indice = encuesta_binaria.index.get_loc(MI_ALIAS)
mi_semestre = semestre_por_alias.iloc[mi_indice]

#%% Matriz de distancias entre todos los encuestados
# Usamos la distancia de emparejamiento (matching), que es el porcentaje de preguntas en las que 2
# personas respondieron diferente. Entre menor sea, más similares son; entre mayor, más diferentes.
D_encuesta = scipy_spatial_distance.pdist(encuesta_binaria, 'matching')
D_encuesta = scipy_spatial_distance.squareform(D_encuesta)

#%% Función para encontrar, dentro de un grupo de candidatos, al más similar y al más diferente
def usuario_similar_y_diferente(indices_candidatos, distancias_fila, aliases):
    distancias_candidatos = distancias_fila[indices_candidatos]
    idx_mas_similar = indices_candidatos[np.argmin(distancias_candidatos)]
    idx_mas_diferente = indices_candidatos[np.argmax(distancias_candidatos)]
    return aliases[idx_mas_similar], aliases[idx_mas_diferente]

aliases_encuesta = np.array(encuesta_binaria.index)
distancias_mi_usuario = D_encuesta[mi_indice]

#%% En general: comparando contra todos los demás encuestados (de cualquier semestre)
indices_otros_general = np.array([i for i in range(len(aliases_encuesta)) if i != mi_indice])
similar_general, diferente_general = usuario_similar_y_diferente(
    indices_otros_general, distancias_mi_usuario, aliases_encuesta)

#%% Este semestre: comparando sólo contra quienes respondieron en mi mismo semestre
indices_otros_semestre = np.array(
    [i for i in indices_otros_general if semestre_por_alias.iloc[i] == mi_semestre])
similar_semestre, diferente_semestre = usuario_similar_y_diferente(
    indices_otros_semestre, distancias_mi_usuario, aliases_encuesta)

#%% Resultados
print(f'\nUsuario más similar a mí en general: {similar_general}')
print(f'Usuario más diferente a mí en general: {diferente_general}')
print(f'\nUsuario más similar a mí este semestre ({mi_semestre}): {similar_semestre}')
print(f'Usuario más diferente a mí este semestre ({mi_semestre}): {diferente_semestre}')
