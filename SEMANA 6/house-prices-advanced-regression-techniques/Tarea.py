import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

print("1. Cargando los datos  ")
datos_train = pd.read_csv('train.csv')
datos_test = pd.read_csv('test.csv')

print("\n2. Combinando datos de train y test  ")
test_ids = datos_test['Id']
precios_train = datos_train['SalePrice']
datos_train = datos_train.drop('SalePrice', axis=1)

datos_train['es_train'] = 1
datos_test['es_train'] = 0

datos_combinados = pd.concat([datos_train, datos_test], axis=0)
print(f"Dimensiones del conjunto combinado: {datos_combinados.shape}")

print("\n3. variables importantes seleccionadas")
variables_importantes = [
    'GrLivArea','TotRmsAbvGrd', 'YearBuilt', 'OverallQual', 'GarageArea','FullBath'
]
print("\n4. Preparando los datos ")
X_combinado = datos_combinados[variables_importantes].copy()

print("\n5. Rellenando valores faltantes ")
for columna in X_combinado.columns:
    mediana = X_combinado[columna].median()
    X_combinado[columna] = X_combinado[columna].fillna(mediana)

print("\n6. Separando datos en train y test ")
X_train = X_combinado[datos_combinados['es_train'] == 1]
y_train = precios_train
X_test_final = X_combinado[datos_combinados['es_train'] == 0]

print("\n7. Dividiendo datos de entrenamiento para validación ")
X_train_temp, X_val, y_train_temp, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("\n8. Entrenando el modelo...")
modelo = LinearRegression()
modelo.fit(X_train_temp, y_train_temp)

print("\n Evaluando el modelo  ")
predicciones_val = modelo.predict(X_val)
r2 = r2_score(y_val, predicciones_val)
rmse_val = math.sqrt(mean_squared_error(y_val, predicciones_val))
mae_val = mean_absolute_error(y_val, predicciones_val)
print(f"\nPrecisión del modelo (R²): {r2:.4f}")
print(f"RMSE (validación): {rmse_val:.2f}")
print(f"MAE (validación): {mae_val:.2f}")

print("\n Importancia de cada variable:")
for variable, importancia in zip(variables_importantes, modelo.coef_):
    print(f"{variable}: {importancia:.2f}")

print("\n Reentrenando modelo con todos los datos de entrenamiento ")
modelo_final = LinearRegression()
modelo_final.fit(X_train, y_train)

print("\n Validación cruzada sobre todo el conjunto de entrenamiento")
cv_scores_mse = cross_val_score(LinearRegression(), X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = [math.sqrt(-s) for s in cv_scores_mse]
print(f"RMSE CV (5 folds) - promedio: {np.mean(cv_rmse):.2f}, std: {np.std(cv_rmse):.2f}")

print("\n  Generando predicciones finales ")
predicciones_finales = modelo_final.predict(X_test_final)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predicciones_finales
})
submission.to_csv('submission.csv', index=False)
print("\nArchivo de submission generado correctamente.")