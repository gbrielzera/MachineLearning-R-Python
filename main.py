import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregamento dos dados
df = pd.read_csv('heart.csv', delimiter=',', skipinitialspace=True)

# 2. Preparação dos dados
x = df.drop(['doenca_cardiaca'], axis=1)
y = df.doenca_cardiaca.values

# Padronização
scaler = StandardScaler().fit(x)
standardX = scaler.transform(x)

# Divisão treino e teste (80/20)
x_train, x_test, y_train, y_test = train_test_split(standardX, y, test_size=0.2, random_state=42)

# 3. Ajuste de Hiperparâmetros
param_grid_nb = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

# Uso de GridSearchCV com Validação Cruzada (cv=5)
grid_nb = GridSearchCV(estimator=GaussianNB(), 
                       param_grid=param_grid_nb, 
                       verbose=0, 
                       cv=5, 
                       scoring='accuracy')

grid_nb.fit(x_train, y_train)

# Definição do melhor modelo encontrado
melhor_nb = grid_nb.best_estimator_

# 4. Avaliação e Previsões
previsoes = melhor_nb.predict(x_test)
acuracia_teste = (melhor_nb.score(x_test, y_test) * 100)
matriz = confusion_matrix(y_test, previsoes)

# 5. Visualização Gráfica (Matriz de Confusão)
plt.figure(figsize=(6,4))
sns.heatmap(matriz,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Saudável', 'Doente'],
            yticklabels=['Saudável', 'Doente'])

plt.xlabel('Previsto pela IA')
plt.ylabel('Realidade (Médico)')
plt.title('Matriz de Confusão: Naive Bayes Otimizado')
plt.show()

# 6. Impressão Formatada dos Resultados
print("\n" + "="*60)
print(f"{'RELATÓRIO TÉCNICO: NAIVE BAYES (PYTHON)':^60}")
print("="*60)

print(f"\n[CONFIGURAÇÃO DO MODELO]")
print(f"Melhor parâmetro 'var_smoothing' encontrado: {grid_nb.best_params_['var_smoothing']}")
print(f"Acurácia média na Validação Cruzada: {grid_nb.best_score_:.4f}")

print(f"\n[DESEMPENHO NO TESTE]")
print(f"Acurácia Geral: {acuracia_teste:.2f}%")

# Matriz de Confusão Detalhada
print("\nMatriz de Confusão:")
df_cm = pd.DataFrame(matriz, 
                     index=['Real: Saudável', 'Real: Doente'], 
                     columns=['Previsto: Saudável', 'Previsto: Doente'])
print("-" * 60)
print(df_cm)
print("-" * 60)

# Métricas de Classificação (Precisão, Revocação e F1-score)
print("\nRelatório de Métricas Detalhadas:")
print(classification_report(y_test, previsoes, target_names=['Saudável (0)', 'Doente (1)']))
print("="*60)