import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Carregamento dos dados
df = pd.read_csv('heart.csv', delimiter=',', skipinitialspace=True)

# Preparação dos dados
x = df.drop(['doenca_cardiaca'], axis=1)
y = df.doenca_cardiaca.values

scaler = StandardScaler().fit(x)
standardX = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(standardX, y, test_size=0.2, random_state=42)

# Modelo Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)

resultado = (nb.score(x_test, y_test) * 100)
previsoes = nb.predict(x_test)

#Visualização Gráfica
matriz = confusion_matrix(y_test, previsoes)

plt.figure(figsize=(6,4))
sns.heatmap(matriz,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Sem Doença', 'Com Doença'],
            yticklabels=['Sem Doença', 'Com Doença'])

plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Naive Bayes')
plt.show()

#Impressão Formatada

print("\n" + "="*50)
print(f"{'RELATÓRIO DE DESEMPENHO: NAIVE BAYES':^50}")
print("="*50)

# Acurácia
print(f"\nAcurácia Geral: {resultado:.2f}%")

# Matriz de Confusão em formato de DataFrame para melhor leitura
print("\nMatriz de Confusão (Detalhada):")
df_cm = pd.DataFrame(matriz, 
                     index=['Real: Saudável', 'Real: Doente'], 
                     columns=['Previsto: Saudável', 'Previsto: Doente'])
print("-" * 50)
print(df_cm)
print("-" * 50)

# Relatório de Classificação
print("\nMétricas de Classificação:")
relatorio_str = classification_report(y_test, previsoes, target_names=['Saudável (0)', 'Doente (1)'])
print(relatorio_str)
print("="*50)