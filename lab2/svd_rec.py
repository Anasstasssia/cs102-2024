import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""
    ratings = df.rename(columns={'Book-Rating': 'Rating'})
    ratings['Rating'] = ratings['Rating'].astype(float)
    ratings = ratings.query("Rating != 0.0")
    min_ratings = 2
    book_counts = ratings.groupby('ISBN')['User-ID'].nunique()
    user_counts = ratings.groupby('User-ID')['ISBN'].nunique()
    good_books = book_counts[book_counts >= min_ratings].index
    good_users = user_counts[user_counts >= min_ratings].index
    filtered1_ratings = ratings[(ratings['ISBN'].isin(good_books)) & (ratings['User-ID'].isin(good_users))]
    return filtered1_ratings


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Rating']], reader)
    train_set, test_set = train_test_split(data, test_size=0.3)
    svd = SVD(n_factors=50, n_epochs=30, lr_all = 0.005, reg_all = 0.1)
    svd.fit(train_set)
    predictions = svd.test(test_set)
    predictions = svd.test(test_set)
    mae = accuracy.mae(predictions)

    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)


# test = pd.read_csv("Ratings.csv")
# test_1 = ratings_preprocessing(test)
# modeling(test_1)
