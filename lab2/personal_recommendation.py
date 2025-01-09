import pandas as pd
import pickle
import warnings
import numpy as np
from scipy.sparse import hstack
from surprise import Dataset, SVD, Reader
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")
ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv")


def shift_row_right(row, start_p, col):
    for i in range(len(col) - 1, start_p, -1):
        row[col[i]] = row[col[i - 1]]
    row[col[start_p]] = np.nan
    return row


invalid_rows = books[books["Year-Of-Publication"].str.match("[^0-9]", na=False)]
columns = books.columns.tolist()
start = columns.index("Book-Author")
books.loc[invalid_rows.index] = books.loc[invalid_rows.index].apply(shift_row_right, axis=1, start_p=start, col=columns)
books = books.iloc[:, :-3]
books = books[books["ISBN"].isin(ratings["ISBN"])]
rating_counts = ratings["User-ID"].value_counts()
rating_not_one = rating_counts[rating_counts != 1].index
ratings = ratings[ratings["User-ID"].isin(rating_not_one)]
user_with_max_zeros = (ratings[ratings["Book-Rating"] == 0]["User-ID"].value_counts().idxmax())
reader = Reader(rating_scale=(1, 10))
non_zero_ratings = ratings[ratings["Book-Rating"] != 0]
data = Dataset.load_from_df(non_zero_ratings[["User-ID", "ISBN", "Book-Rating"]], reader)
train = data.build_full_trainset()
svd = SVD()
svd.fit(train)

user_ratings = ratings[ratings["User-ID"] == user_with_max_zeros]
zero_rated_books = user_ratings[user_ratings["Book-Rating"] == 0]["ISBN"].tolist()
recommendations = []
for isbn in zero_rated_books:
    pr = svd.predict(user_with_max_zeros, isbn)
    if pr.est >= 8:
        recommendations.append((isbn, pr.est))
average_ratings = ratings.groupby("ISBN")["Book-Rating"].mean().reset_index()
average_ratings.rename(columns={"Book-Rating": "Average-Rating"}, inplace=True)
final_data = pd.merge(books, average_ratings, on="ISBN", how="left").dropna(subset=["Average-Rating"])
X = final_data[["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]]
y = final_data["Average-Rating"]
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_title = vectorizer.fit_transform(X["Book-Title"])
label_author = LabelEncoder()
label_publisher = LabelEncoder()
X_author = label_author.fit_transform(X["Book-Author"]).reshape(-1, 1)
X_publisher = label_publisher.fit_transform(X["Publisher"]).reshape(-1, 1)
scaler_year = StandardScaler()
X_year = scaler_year.fit_transform(X[["Year-Of-Publication"]])
scaler_author = StandardScaler()
scaler_publisher = StandardScaler()
X_author_scaled = scaler_author.fit_transform(X_author)
X_publisher = scaler_publisher.fit_transform(X_publisher)
X_combined = hstack([X_title, X_author_scaled, X_publisher, X_year])
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_scaled, test_size=0.1, random_state=42)
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train.ravel())
recommended_isbn = [isbn for isbn, _ in recommendations]
filtered_books = final_data[final_data["ISBN"].isin(recommended_isbn)].drop_duplicates(subset=["ISBN"])
X_filtered_title = vectorizer.transform(filtered_books["Book-Title"])
X_filtered_author = label_author.transform(filtered_books["Book-Author"]).reshape(-1, 1)
X_filtered_publisher = label_publisher.transform(filtered_books["Publisher"]).reshape(-1, 1)
X_filtered_year = scaler_year.transform(filtered_books[["Year-Of-Publication"]])
X_filtered_author_scaled = scaler_author.transform(X_filtered_author)
X_filtered_publisher_scaled = scaler_publisher.transform(X_filtered_publisher)
X_filtered_combined = hstack([X_filtered_title, X_filtered_author_scaled, X_filtered_publisher_scaled, X_filtered_year])
lin_reg_predictions = model.predict(X_filtered_combined)
lin_reg_predictions_unscaled = scaler_y.inverse_transform(lin_reg_predictions.reshape(-1, 1))
filtered_books["LinReg-Pr"] = lin_reg_predictions_unscaled
filtered_books["SVD_Pr"] = [pr for isbn, pr in recommendations if isbn in filtered_books["ISBN"].tolist()]
final_recommendations = filtered_books.sort_values(by=["LinReg-Pr"], ascending=False)
print(final_recommendations[["Book-Title", "LinReg-Pr", "SVD_Pr"]])

# Запись рекомендаций в текстовый файл
output_file = "personal_recommendation_text_file.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Personal Book Recommendations:\n\n")
    for _, row in final_recommendations.iterrows():
        f.write(f"Title: {row['Book-Title']}\n")
        f.write(f"Linear Regression Prediction: {row['LinReg-Pr']:.2f}\n")
        f.write(f"SVD Prediction: {row['SVD_Pr']:.2f}\n")
        f.write("-" * 40 + "\n")

# Сохранение моделей
with open("model3.pkl", "wb") as f:
    pickle.dump(svd, f)
with open("model4.pkl", "wb") as f:
    pickle.dump(model, f)