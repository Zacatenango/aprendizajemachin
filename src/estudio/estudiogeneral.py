import pandas
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split

# Paso 1: MSE y R cuadrada
y = [3,5,7,9]
y_gorrito = [9,7,5,3]

MSE = sum((una_y - una_y_gorrito)**2 for una_y, una_y_gorrito in zip(y, y_gorrito)) / len(y)
print(MSE)

promedio = sum(y)/len(y)
MSE_promedio = sum((una_y - promedio)**2 for una_y in y)/len(y)
R_cuadrada = 1 - (MSE/MSE_promedio)
print(R_cuadrada)

#########
df = pandas.read_csv("HW2_training.csv")
print(df.columns.tolist())
print()
print(df.dtypes)
print()
print(df.head(3))
print()
print(df.isna().sum())

df.columns = df.columns.str.strip()  # Quitamos espacios en columnas
df = df.drop(columns=["Unnamed: 10"])  # Tumbamos columna vacía

def convertir_str_a_float(_param_str_numero):
   # Bakayoke: ¿no es un string? de ser así, aborto
   if type(_param_str_numero) is not str: return _param_str_numero
   # Caso 1: espacio en blanco alrededor
   _param_str_numero = _param_str_numero.strip()
   # Caso 2: quitar comas
   _param_str_numero = _param_str_numero.replace(",", "")
   # Caso 3: ¿hay paréntesis? cambio por un -
   if ( _param_str_numero.find('(') != -1 ):
      _param_str_numero = _param_str_numero.replace('(', '-')
      _param_str_numero = _param_str_numero.replace(')', '')
   # Convierto a float; si todavía no puedo después de este proceso
   # de conversión, me rindo y tiro None
   try:
      resultao = float(_param_str_numero)
   except ValueError:
      resultao = None
   return resultao

print(convertir_str_a_float('609,366.9835'))
print(convertir_str_a_float('(158,355.4276)'))
print(convertir_str_a_float('0.3745401190'))
print(convertir_str_a_float(''))

df = df.map(convertir_str_a_float)

print("------- Revisión ---------")
print(df.dtypes)
print()
print(df.isna().sum())
print()
print(df.shape)

###############
# División entrenamiento-prueba
# Descubrí que el final de mis datos no tiene Y. Necesito la Y para poder entrenar, por lo que tumbo
# las filas que no tienen Y
df_holdout = df[df['Y'].isna()].copy()
df = df.dropna(subset=['Y'])
print(df.shape)
print(df.isna().sum())

X = df.drop("Y", axis=1)  # Para saber cuál eje es cual, doy df.shape, y la tupla resultante es (<cantidad en el eje 0>, <cantidad en el eje 1>, ...)
Y = df["Y"]
# 20% Test, 80% Entrenamiento
X_entreno, X_prueba, Y_entreno, Y_prueba = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_entreno.shape, X_prueba.shape)

print(len(X_entreno))
print(len(X_entreno.dropna()))

# Calidad de datos
# .T me saca la transpuesta
print(df.describe().T)

# Para ver esto más fácil, grafico
features = ['X0','X1','X2','X3','X4','X5','X6','X7','X8']

fig, axes = pyplot.subplots(3, 3, figsize=(14, 10))
for ax, col in zip(axes.ravel(), features):
    df[col].dropna().hist(bins=50, ax=ax)
    ax.set_title(col)
pyplot.tight_layout()
pyplot.show()

# Puedo ver ahí que X0, X1, X2, X6, X7 y X8 tienen la región de datos toda comprimida. Esto sucede
# porque estas gráficas son histogramas, los cuales grafican todo el rango de datos; si tengo
# outliers extremos (tal vez puestos ahí adrede por el prof), esto hará que la región donde sí hay
# valores esté comprimida. Cuando eso sucede, está la opción de hacer zoom en el eje X.
# Debido a que la API moderna de pyplot es totalmente alien para mí, mejor lo voy a hacer estilo
# MATLAB.
features = ['X0','X1','X2','X6','X7','X8']
pyplot.figure(figsize=(11,8.5))  # Tamaño carta acostado, la dimensión es en pulgadas
for indice, columna in enumerate(features):
   pyplot.subplot(2,3,indice+1)
   datos = df[columna].dropna()
   minimo, maximo = datos.quantile([0.1, 0.9])
   pyplot.hist(datos[(datos >= minimo) & (datos <= maximo)], bins=50)
   pyplot.title(columna)
pyplot.tight_layout()
pyplot.show()

# Ahora revisamos la correlación de nuestras columnas con Y.
print(df.corr(numeric_only=True)['Y'].sort_values())
