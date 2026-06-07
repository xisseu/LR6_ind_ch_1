# ============================================================================
# СТРОКИ 1-6: ЗАКОММЕНТИРОВАННЫЙ ТЕСТОВЫЙ КОД (ИСХОДНЫЙ)
# ============================================================================
# import matplotlib.pyplot as plt
# fig,ax= plt. subplots()
# ax.plot([1,2,3,4],[1,4,2,5])
# plt.ylabel('some numbers')
# plt.savefig('myfig.png')

# ============================================================================
# БЛОК 1: ИМПОРТ НЕОБХОДИМЫХ БИБЛИОТЕК
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# БЛОК 2: ГЕНЕРАЦИЯ ДАННЫХ ПО ЗАДАННОМУ ВАРИАНТУ
# ============================================================================
# Установка seed для воспроизводимости результатов
np.random.seed(42)

# Количество записей
n_samples = 200

# Генерация первого параметра: диапазон [0.01; 1]
param1 = np.random.uniform(0.01, 1.0, n_samples)

# Генерация второго параметра: диапазон [1; 300]
param2 = np.random.uniform(1, 300, n_samples)

# Генерация третьего параметра: категориальные значения (Самара, Тольятти, Москва)
cities = ['Самара', 'Тольятти', 'Москва']
param3 = np.random.choice(cities, n_samples, p=[0.3, 0.3, 0.4])  # вероятности для распределения

# Создание DataFrame с исходными данными
df_original = pd.DataFrame({
    'Параметр_1': param1,
    'Параметр_2': param2,
    'Параметр_3': param3
})

print("=" * 80)
print("ИСХОДНЫЕ ДАННЫЕ (первые 10 строк):")
print("=" * 80)
print(df_original.head(10))
print("\nИнформация о данных:")
print(df_original.info())

# ============================================================================
# БЛОК 3: ПРЕОБРАЗОВАНИЕ ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ
# ============================================================================
# Преобразование категориального признака (Параметр_3) в числовой с помощью LabelEncoder
# Это необходимо, так как KMeans работает только с числовыми данными
label_encoder = LabelEncoder()
df_original['Параметр_3_encoded'] = label_encoder.fit_transform(df_original['Параметр_3'])
# Соответствие кодов городам: Самара -> 0, Тольятти -> 1, Москва -> 2
city_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nСоответствие городов числовым кодам:")
print(city_mapping)

# Подготовка матрицы признаков X для кластеризации (3 параметра)
X = df_original[['Параметр_1', 'Параметр_2', 'Параметр_3_encoded']].values

# Стандартизация данных (масштабирование) для корректной работы KMeans
# Это важно, так как параметры имеют разные диапазоны: [0.01-1], [1-300], [0-2]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nСтатистика после стандартизации:")
print(f"Среднее каждого признака: {X_scaled.mean(axis=0)}")
print(f"Стандартное отклонение каждого признака: {X_scaled.std(axis=0)}")

# ============================================================================
# БЛОК 4: ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ (МЕТОД ЛОКТЯ)
# ============================================================================
# Вычисление инерции для различного количества кластеров (от 1 до 10)
inertias = []
silhouette_scores = []
K_range = range(2, 11)  # от 2 до 10 (силуэт не определен для k=1)

print("\n" + "=" * 80)
print("ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ:")
print("=" * 80)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))
    print(f"k={k}: Инерция={kmeans_temp.inertia_:.2f}, Силуэт={silhouette_scores[-1]:.4f}")

# Визуализация метода локтя
plt.figure(figsize=(12, 5))

# График 1: Метод локтя (инерция)
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Количество кластеров (k)', fontsize=12)
plt.ylabel('Инерция (WCSS)', fontsize=12)
plt.title('Метод локтя для определения оптимального k', fontsize=14)
plt.grid(True, alpha=0.3)
# Отмечаем точку локтя (k=3 или k=4, в зависимости от данных)
# По умолчанию выбираем k=4, но можно проанализировать
optimal_k_elbow = 4  # предположительное значение

# График 2: Силуэтный коэффициент
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Количество кластеров (k)', fontsize=12)
plt.ylabel('Силуэтный коэффициент', fontsize=12)
plt.title('Силуэтный коэффициент для различных k', fontsize=14)
plt.grid(True, alpha=0.3)

# Определение оптимального k по максимальному силуэту
optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
print(f"\nОптимальное k по методу локтя (визуально): ~{optimal_k_elbow}")
print(f"Оптимальное k по силуэтному коэффициенту: {optimal_k_silhouette}")

optimal_k = optimal_k_silhouette  # выбираем оптимальное по силуэту
print(f"Выбрано оптимальное количество кластеров: k = {optimal_k}")

plt.tight_layout()
plt.savefig('elbow_silhouette_analysis.png', dpi=150)
print("\nГрафик анализа сохранен как 'elbow_silhouette_analysis.png'")

# ============================================================================
# БЛОК 5: ОБУЧЕНИЕ МОДЕЛИ KMEANS С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ
# ============================================================================
print("\n" + "=" * 80)
print("ОБУЧЕНИЕ МОДЕЛИ KMEANS:")
print("=" * 80)

# Создание и обучение модели KMeans
kmeans = KMeans(
    n_clusters=optimal_k,        # оптимальное количество кластеров
    init='k-means++',            # умная инициализация центроидов
    random_state=42,             # фиксация случайности
    n_init=10,                   # количество запусков с разными центроидами
    max_iter=300                 # максимальное количество итераций
)

# Обучение модели на стандартизированных данных
kmeans.fit(X_scaled)

# Получение меток кластеров для каждой точки
cluster_labels = kmeans.labels_

# Получение центроидов в стандартизированном пространстве
centroids_scaled = kmeans.cluster_centers_

# Обратное преобразование центроидов в исходный масштаб для интерпретации
centroids_original = scaler.inverse_transform(centroids_scaled)

# ============================================================================
# БЛОК 6: ВЫВОД РЕЗУЛЬТАТОВ В КОНСОЛЬ
# ============================================================================
print("\n" + "=" * 80)
print("РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ:")
print("=" * 80)

# 6.1. Вывод набора данных с распределением по кластерам
df_results = df_original.copy()
df_results['Прогнозируемый_кластер'] = cluster_labels
df_results['Параметр_3_оригинал'] = df_original['Параметр_3']

print("\n1. НАБОР ДАННЫХ С РАСПРЕДЕЛЕНИЕМ ПО КЛАСТЕРАМ (первые 20 записей):")
print("-" * 80)
print(df_results[['Параметр_1', 'Параметр_2', 'Параметр_3_оригинал', 'Прогнозируемый_кластер']].head(20))

# 6.2. Прогнозируемые кластеры для каждой точки данных (первые 50)
print("\n2. ПРОГНОЗИРУЕМЫЕ КЛАСТЕРЫ ДЛЯ КАЖДОЙ ТОЧКИ ДАННЫХ (первые 50):")
print("-" * 80)
print(f"Метки кластеров: {cluster_labels[:50]}")

# 6.3. Координаты центроидов для каждого кластера
print("\n3. КООРДИНАТЫ ЦЕНТРОИДОВ ДЛЯ КАЖДОГО КЛАСТЕРА:")
print("-" * 80)
for i in range(optimal_k):
    print(f"Кластер {i}:")
    print(f"  - Параметр_1 (нормализованный): {centroids_scaled[i][0]:.4f} → исходный: {centroids_original[i][0]:.4f}")
    print(f"  - Параметр_2 (нормализованный): {centroids_scaled[i][1]:.4f} → исходный: {centroids_original[i][1]:.2f}")
    print(f"  - Параметр_3 (нормализованный): {centroids_scaled[i][2]:.4f} → исходный: {centroids_original[i][2]:.2f} (код города)")

    # Декодирование города (приблизительное, так как центроид может быть между значениями)
    city_code_rounded = round(centroids_original[i][2])
    if city_code_rounded in city_mapping.values():
        inv_map = {v: k for k, v in city_mapping.items()}
        print(f"  - Соответствующий город: {inv_map.get(city_code_rounded, 'не определен')}")

# 6.4. Внутрикластерная сумма квадратов
print(f"\n4. ВНУТРИКЛАСТЕРНАЯ СУММА КВАДРАТОВ (ИНЕРЦИЯ):")
print("-" * 80)
print(f"Инерция (WCSS) = {kmeans.inertia_:.4f}")
print("(Сумма квадратов расстояний от каждой точки до центроида своего кластера)")

# 6.5. Количество итераций
print(f"\n5. КОЛИЧЕСТВО ИТЕРАЦИЙ АЛГОРИТМА:")
print("-" * 80)
print(f"Количество итераций для достижения сходимости: {kmeans.n_iter_}")
print("(Чем меньше итераций, тем быстрее сошелся алгоритм)")

# 6.6. Размер каждого кластера
print(f"\n6. РАЗМЕР КАЖДОГО КЛАСТЕРА:")
print("-" * 80)
cluster_sizes = Counter(cluster_labels)
for cluster_id in sorted(cluster_sizes.keys()):
    size = cluster_sizes[cluster_id]
    percentage = (size / n_samples) * 100
    print(f"Кластер {cluster_id}: {size} точек ({percentage:.1f}% от всех данных)")

# ============================================================================
# БЛОК 7: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ
# ============================================================================
print("\n" + "=" * 80)
print("ПОСТРОЕНИЕ ВИЗУАЛИЗАЦИЙ:")
print("=" * 80)

# Создание фигуры с несколькими подграфиками
fig = plt.figure(figsize=(16, 12))

# 7.1. Диаграмма рассеяния: Параметр_1 vs Параметр_2
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
scatter = ax1.scatter(df_results['Параметр_1'], 
                       df_results['Параметр_2'], 
                       df_results['Параметр_3_encoded'],
                       c=cluster_labels, 
                       cmap='viridis', 
                       s=50, 
                       alpha=0.7)
ax1.set_xlabel('Параметр 1 [0.01-1]', fontsize=10)
ax1.set_ylabel('Параметр 2 [1-300]', fontsize=10)
ax1.set_zlabel('Параметр 3 (код города)', fontsize=10)
ax1.set_title('3D Визуализация кластеров', fontsize=12)
plt.colorbar(scatter, ax=ax1, label='Номер кластера')

# 7.2. Диаграмма рассеяния: Параметр_1 vs Параметр_2 (2D)
ax2 = fig.add_subplot(2, 2, 2)
scatter2 = ax2.scatter(df_results['Параметр_1'], 
                        df_results['Параметр_2'], 
                        c=cluster_labels, 
                        cmap='plasma', 
                        s=50, 
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5)
# Добавление центроидов (обратное преобразование в масштаб параметра 1 и 2)
ax2.scatter(centroids_original[:, 0], centroids_original[:, 1], 
            marker='X', c='red', s=200, label='Центроиды', edgecolors='darkred', linewidth=2)
ax2.set_xlabel('Параметр 1 (диапазон [0.01; 1])', fontsize=11)
ax2.set_ylabel('Параметр 2 (диапазон [1; 300])', fontsize=11)
ax2.set_title('Кластеризация: Параметр 1 vs Параметр 2', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Номер кластера')

# 7.3. Диаграмма рассеяния: Параметр_1 vs Параметр_3
ax3 = fig.add_subplot(2, 2, 3)
# Создание цветовой карты для городов
city_colors = {'Самара': 'blue', 'Тольятти': 'green', 'Москва': 'red'}
city_color_list = [city_colors[city] for city in df_results['Параметр_3_оригинал']]
scatter3 = ax3.scatter(df_results['Параметр_1'], 
                        df_results['Параметр_3_оригинал'],
                        c=cluster_labels,
                        cmap='coolwarm',
                        s=50,
                        alpha=0.7)
ax3.set_xlabel('Параметр 1 [0.01-1]', fontsize=11)
ax3.set_ylabel('Параметр 3 (Город)', fontsize=11)
ax3.set_title('Кластеризация: Параметр 1 vs Город', fontsize=12)
plt.colorbar(scatter3, ax=ax3, label='Номер кластера')

# 7.4. Диаграмма рассеяния: Параметр_2 vs Параметр_3
ax4 = fig.add_subplot(2, 2, 4)
scatter4 = ax4.scatter(df_results['Параметр_2'], 
                        df_results['Параметр_3_оригинал'],
                        c=cluster_labels,
                        cmap='Spectral',
                        s=50,
                        alpha=0.7)
ax4.set_xlabel('Параметр 2 [1-300]', fontsize=11)
ax4.set_ylabel('Параметр 3 (Город)', fontsize=11)
ax4.set_title('Кластеризация: Параметр 2 vs Город', fontsize=12)
plt.colorbar(scatter4, ax=ax4, label='Номер кластера')

plt.suptitle('ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ K-MEANS', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('clustering_visualization.png', dpi=150)
print("График визуализации сохранен как 'clustering_visualization.png'")

# 7.5. Дополнительная визуализация: распределение параметров по кластерам
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Распределение параметра 1 по кластерам
ax = axes[0, 0]
for i in range(optimal_k):
    cluster_data = df_results[df_results['Прогнозируемый_кластер'] == i]
    ax.hist(cluster_data['Параметр_1'], bins=15, alpha=0.5, label=f'Кластер {i}')
ax.set_xlabel('Параметр 1', fontsize=11)
ax.set_ylabel('Частота', fontsize=11)
ax.set_title('Распределение Параметра 1 по кластерам', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Распределение параметра 2 по кластерам
ax = axes[0, 1]
for i in range(optimal_k):
    cluster_data = df_results[df_results['Прогнозируемый_кластер'] == i]
    ax.hist(cluster_data['Параметр_2'], bins=15, alpha=0.5, label=f'Кластер {i}')
ax.set_xlabel('Параметр 2', fontsize=11)
ax.set_ylabel('Частота', fontsize=11)
ax.set_title('Распределение Параметра 2 по кластерам', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Распределение городов по кластерам
ax = axes[1, 0]
city_cluster_crosstab = pd.crosstab(df_results['Параметр_3_оригинал'], df_results['Прогнозируемый_кластер'])
city_cluster_crosstab.plot(kind='bar', ax=ax, colormap='viridis')
ax.set_xlabel('Город', fontsize=11)
ax.set_ylabel('Количество точек', fontsize=11)
ax.set_title('Распределение городов по кластерам', fontsize=12)
ax.legend(title='Кластер')
ax.grid(True, alpha=0.3)

# Box plot для параметров по кластерам
ax = axes[1, 1]
df_melted = df_results.melt(id_vars=['Прогнозируемый_кластер'], 
                             value_vars=['Параметр_1', 'Параметр_2'],
                             var_name='Параметр', 
                             value_name='Значение')
sns.boxplot(x='Прогнозируемый_кластер', y='Значение', hue='Параметр', data=df_melted, ax=ax)
ax.set_title('Box-plot распределения параметров по кластерам', fontsize=12)
ax.set_xlabel('Номер кластера', fontsize=11)
ax.set_ylabel('Значение параметра', fontsize=11)
ax.legend(title='Параметр')
ax.grid(True, alpha=0.3)

plt.suptitle('ДЕТАЛЬНЫЙ АНАЛИЗ КЛАСТЕРОВ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_analysis.png', dpi=150)
print("График детального анализа сохранен как 'cluster_analysis.png'")

# Отображение всех графиков
plt.show()

print("\n" + "=" * 80)
print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
print("=" * 80)