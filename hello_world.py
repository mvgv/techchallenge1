import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    print("--- Scikit-Learn 'Hello World' ---")
    
    # 1. Preparando os dados (Features e Labels)
    # Vamos criar um padrão simples: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]]) # Features (entradas)
    y = np.array([3, 5, 7, 9, 11])          # Labels (saídas esperadas)

    # 2. Escolhendo e instanciando o modelo
    # LinearRegression é um dos modelos mais simples de Machine Learning
    model = LinearRegression()

    # 3. Treinando o modelo (ajustando a linha aos dados)
    print("Treinando o modelo com os dados (y = 2x + 1)...")
    model.fit(X, y)

    # 4. Fazendo uma previsão com um dado novo
    # Vamos pedir para o modelo prever o resultado quando x for igual a 6.
    # O esperado é 13 (pois 2*6 + 1 = 13).
    test_value = np.array([[6]])
    prediction = model.predict(test_value)

    print(f"\nPrevisão para x=6: {prediction[0]:.2f}")
    print("Sucesso! O modelo aprendeu o padrão e fez a previsão corretamente.")

if __name__ == "__main__":
    main()
