import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss


def print_dataset(data, title='Часовий ряд'):
    plt.figure(figsize=(12, 8))
    plt.plot(data['Month'], data['#Passengers'], marker='o')
    plt.title(title)
    plt.xlabel('Місяць і рік')
    plt.ylabel('К-сть пасажирів')
    plt.xticks(np.arange(0, len(data['Month']), step=8))
    plt.tight_layout()
    plt.show()


def check_miss_val(data):
    print("\tПропуски в даних:")
    missing_values = data.isnull().sum()
    print(missing_values.to_string())


def create_miss_val(data):
    np.random.seed()
    random_indices = np.random.choice(data.index, size=10, replace=False)  # Вибір 10 випадкових індексів
    data_with_missing = data.copy()  # Копія датасету з пропущеними значеннями
    data_with_missing.loc[random_indices, '#Passengers'] = np.nan  # Позначення обраних рядків як пропущених
    return data_with_missing


def fill_miss_val(data_with_missing):
    median_filled_data = data_with_missing['#Passengers'].fillna(data_with_missing['#Passengers'].median())
    mean_filled_data = data_with_missing['#Passengers'].fillna(data_with_missing['#Passengers'].mean())
    return median_filled_data, mean_filled_data


def print_graphics(data_with_missing, median_filled_data, mean_filled_data):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(data_with_missing['Month'], data_with_missing['#Passengers'], marker='o')
    plt.title('Часовий ряд із штучно створеними пропусками')
    plt.xlabel('Місяць')
    plt.ylabel('К-сть пасажирів')
    plt.xticks(np.arange(0, len(data_with_missing['Month']), step=8))
    plt.subplot(2, 1, 2)
    plt.plot(data_with_missing['Month'], median_filled_data, marker='o', label='заповнення медіаною')
    plt.plot(data_with_missing['Month'], mean_filled_data, marker='o', color='orange', label='заповнення середнім '
                                                                                             'арифметичним значенням')
    plt.title('Часовий ряд після заповнення пропусків')
    plt.xlabel('Місяць')
    plt.ylabel('К-сть пасажирів')
    plt.xticks(np.arange(0, len(data_with_missing['Month']), step=8))
    plt.legend()
    plt.tight_layout()
    plt.show()


def moving_average(data):
    window_size = 4
    ma_column_name = f'MA_{window_size}'

    # Розрахунок ковзного середнього
    data[ma_column_name] = data['#Passengers'].rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(data['Month'], data['#Passengers'], label='Реальне значення', marker='o')
    plt.plot(data['Month'], data[ma_column_name], label=f'Фільтроване значення (метод ковзного середнього із '
                                                        f'розміром вікна = {window_size} місяці', color='orange')
    plt.xticks(np.arange(0, len(data['Month']), step=8))
    plt.xlabel('Місяць і рік')
    plt.ylabel('К-сть пасажирів')
    plt.legend()
    plt.show()

    data['#Passengers'] = data[ma_column_name]
    del data[ma_column_name]
    return data


def check_stationarity(t_data):
    print("\n\tПеревірка на стаціонарність:")
    num_intervals = 6
    interval_size = len(t_data) // num_intervals

    mean_values = []
    var_values = []

    for i in range(num_intervals):
        start_idx = i * interval_size
        end_idx = (i + 1) * interval_size
        interval_data = t_data.iloc[start_idx:end_idx]
        mean_values.append(round(interval_data['#Passengers'].mean(), 2))
        var_values.append(round(interval_data['#Passengers'].var(), 2))

    for i in range(num_intervals):
        start_month = 1 + i * interval_size
        end_month = (i + 1) * interval_size
        print(f"\tІнтервал {i + 1} ({start_month}m - {end_month}m):")
        print(f"Середнє значення: {mean_values[i]}")
        print(f"Дисперсія: {var_values[i]}\n")


def trend_time_series_linear_regression(t_data):
    t_data = t_data.copy()
    t_data[['Year', 'Month']] = t_data['Month'].str.split('-', expand=True)

    t_data['Year'] = pd.to_numeric(t_data['Year'])
    t_data['Passengers'] = pd.to_numeric(t_data['#Passengers'])

    sum_passengers = t_data.groupby('Year')['Passengers'].sum().reset_index()

    coefficients = np.polyfit(sum_passengers['Year'], sum_passengers['Passengers'], 1)
    slope, intercept = coefficients

    x = sum_passengers['Year']
    y1 = sum_passengers['Passengers']

    plt.plot(x, y1)
    plt.plot()

    plt.xlabel("Рік")
    plt.ylabel("К-сть пасажирів")
    plt.title("Тренд")
    plt.show()

    # Коефіцієнт нахилу показує швидкість зміни тренду
    print(f"Коефіцієнт нахилу (швидкість зміни тренду): {slope:.2f} пасажирів на рік")


def adf_test(data):
    test_data = data
    print("\n\tADF-тест (Augmented Dickey-Fuller test):")
    result = adfuller(test_data['#Passengers'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    print('Lags Used:', result[2])
    print('Number of Observations Used:', result[3])

    print("\tРезультат тесту:")
    if result[1] <= 0.05:
        print("Ряд є стаціонарним. Тренду немає.")
    else:
        print("Ряд не є стаціонарним. Є тренд.")


def kpss_test(data):
    test_data = data
    print("\n\tKPSS-тест (Kwiatkowski-Phillips-Schmidt-Shin test):")
    result = kpss(test_data['#Passengers'])

    # Виведемо результати тесту
    print('KPSS Statistic:', result[0])
    print('p-value:', result[1])
    print('Lags Used:', result[2])
    print('Critical Values:', result[3])

    print("\tРезультат тесту:")
    if result[1] <= 0.05:
        print("Ряд не є стаціонарним. Є тренд.")
    else:
        print("Ряд є стаціонарним. Тренду немає.")


def differencing(data):
    data['#Passengers'] = data['#Passengers'].diff().fillna(0)
    return data


def main():
    data = pd.read_csv("AirPassengers.csv")
    print_dataset(data)

    # Завдання 2.
    check_miss_val(data)
    data_with_missing = create_miss_val(data)
    median_filled_data, mean_filled_data = fill_miss_val(data_with_missing)
    print_graphics(data_with_missing, median_filled_data, mean_filled_data)
    data['#Passengers'] = median_filled_data

    # Завдання 3. Фільтрація
    data = moving_average(data)

    # Завдання 5. Тренд, сезонність, стаціонарність
    check_stationarity(data)
    trend_time_series_linear_regression(data)
    adf_test(data)
    kpss_test(data)
    data = differencing(data)
    print_dataset(data, title='Часовий ряд після диференціювання першого порядку')
    adf_test(data)
    kpss_test(data)


if __name__ == "__main__":
    main()
