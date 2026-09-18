# Paso 1: MSE y R cuadrada
y = [3,5,7,9]
y_gorrito = [9,7,5,3]

MSE = sum((una_y - una_y_gorrito)**2 for una_y, una_y_gorrito in zip(y, y_gorrito)) / len(y)
print(MSE)

promedio = sum(y)/len(y)
MSE_promedio = sum((una_y - promedio)**2 for una_y in y)/len(y)
R_cuadrada = 1 - (MSE/MSE_promedio)
print(R_cuadrada)
