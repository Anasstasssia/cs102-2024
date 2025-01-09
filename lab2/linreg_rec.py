import pickle
import re
import nltk
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

nltk.download("stopwords")
nltk.download("punkt")


def fix_merged_columns(row):
    merged_value = row['Book-Title']
    pattern = r'^(.+?)\\";(.+)"$'
    match = re.match(pattern, merged_value)
    if match:
        author, title = match.groups()
        return author.strip(), title.strip()
    else:
        return None, None


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""
    df = df.copy()

    new_sort = df[df["Year-Of-Publication"].map(str).str.match("[^0-9]")]
    df['Year-Of-Publication'] = df['Year-Of-Publication'].astype(str)
    mask = ~df["Year-Of-Publication"].str.contains(r'\D')
    books_filtered = df[mask]

    # Применяем функцию исправления данных
    new_sort = df[~mask]
    fixed_columns = new_sort.apply(lambda x: fix_merged_columns(x), axis=1)
    new_sort.loc[:, ['Book-Author', 'Book-Title']] = pd.DataFrame(fixed_columns.tolist(), index=new_sort.index)

    df = pd.concat([books_filtered, new_sort])
    # Удаляем строки с будущими годами публикации
    current_year = pd.Timestamp.now().year
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')

    df.loc[(df['Year-Of-Publication'] > current_year), 'Year-Of-Publication'] = 2024
    # Удаление ненужных столбцов с изображениями
    df = df.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

    df['Book-Author'] = df['Book-Author'].fillna('Unknown')
    df['Publisher'] = df['Publisher'].fillna('Unknown')

    return df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""
    df = df.copy()

    # Удаление нулевых рейтингов
    df = df[df['Book-Rating'] != 0]
    # Задаем минимально кол-во
    min_books_ratings = 2

    # Рассчитываем средний рейтинг для каждой книги
    book_avg_ratings = df.groupby('ISBN')['Book-Rating'].mean()

    # Заменяем индивидуальные рейтинги на средние для книги
    df['Book-Rating'] = df['ISBN'].map(book_avg_ratings)

    # Рассчитываем число оценок для каждой книги
    book_ratings_count = df.groupby('ISBN')['Book-Rating'].count()

    # Добавляем количество оценок к исходным данным
    df['Book-Ratings-Count'] = df['ISBN'].map(book_ratings_count)

    # Фильтруем книги, у которых меньше min_books_ratings оценок
    df = df[df['Book-Ratings-Count'] >= min_books_ratings]

    # Удалим пользователей, которые поставили меньше min_books_ratings оценок
    user_ratings_count = df.groupby('User-ID')['Book-Rating'].count()
    good_users = user_ratings_count[user_ratings_count >= min_books_ratings].index
    df = df[df['User-ID'].isin(good_users)]

    return df


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """
    stop_words = set(stopwords.words('english'))
    # Токенизация (разделение на слова)
    tokens = nltk.word_tokenize(text.lower())
    # Удаление стоп-слов и пунктуации
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Бинаризовать или представить в виде чисел категориальные столбцы (кроме названий)
    2. Разбить данные на тренировочную и обучающую выборки
    3. Векторизовать подвыборки и создать датафреймы из векторов (размер вектора названия в тестах – 1000)
    4. Сформировать итоговые X_train, X_test, y_train, y_test
    5. Обучить и протестировать SGDRegressor
    6. Подобрать гиперпараметры (при необходимости)
    7. Сохранить модель"""
    categorical_features = ['Book-Author', 'Publisher']
    numerical_features = ['Year-Of-Publication']
    for feature in categorical_features:
        books[feature] = books[feature].astype('category').cat.codes
    avg_ratings = ratings[['ISBN', 'Book-Rating']].drop_duplicates()
    final_df = books.merge(avg_ratings, on='ISBN', how='inner')
    X = final_df[numerical_features + categorical_features]
    y = final_df['Book-Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    titles = final_df['Book-Title']
    X_titles_train = vectorizer.fit_transform(titles.iloc[X_train.index])
    X_titles_test = vectorizer.transform(titles.iloc[X_test.index])

    X_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(X_titles_train.toarray())], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(X_titles_test.toarray())], axis=1)
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    linreg = SGDRegressor(random_state=10)
    linreg.fit(X_train, y_train)

    # Тестирование модели и вывод метрик
    y_pred = linreg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'MAE: {mae}')
    with open("linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)


# books1 = pd.read_csv("Books.csv")
# ratings1 = pd.read_csv("Ratings.csv")
# filtered_ratings1 = ratings_preprocessing(ratings1)
# filtered_books1 = books_preprocessing(books1)
# 
# modeling(filtered_books1, filtered_ratings1)
