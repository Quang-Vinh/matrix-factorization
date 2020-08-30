# Matrix Factorization
Short and simple implementation of kernel matrix factorization with online-updating for use in collaborative recommender systems built on top of scikit-learn.

## Prerequisites
- Python 3
- numba
- numpy
- pandas
- scikit-learn
- scipy

## Installation
```
pip install matrix_factorization
```

## Usage
```python
from matrix_factorization import BaselineModel, KernelMF, train_update_test_split

import pandas as pd
from sklearn.metrics import mean_squared_error

# Movie data found here https://grouplens.org/datasets/movielens/
cols = ["user_id", "item_id", "rating", "timestamp"]
movie_data = pd.read_csv(
    "../data/ml-100k/u.data", names=cols, sep="\t", usecols=[0, 1, 2], engine="python"
)

X = movie_data[["user_id", "item_id"]]
y = movie_data["rating"]

# Prepare data for online learning
(
    X_train_initial,
    y_train_initial,
    X_train_update,
    y_train_update,
    X_test_update,
    y_test_update,
) = train_update_test_split(movie_data, frac_new_users=0.2)

# Initial training
matrix_fact = KernelMF(n_epochs=20, n_factors=100, verbose=1, lr=0.001, reg=0.005)
matrix_fact.fit(X_train_initial, y_train_initial)

# Update model with new users
matrix_fact.update_users(
    X_train_update, y_train_update, lr=0.001, n_epochs=20, verbose=1
)
pred = matrix_fact.predict(X_test_update)
rmse = mean_squared_error(y_test_update, pred, squared=False)
print(f"\nTest RMSE: {rmse:.4f}")

# Get recommendations
user = 200
items_known = X_train_initial.query("user_id == @user")["item_id"]
matrix_fact.recommend(user=user, items_known=items_known)
```

Check examples/recommender-system.ipynb for complete examples

## License
This project is licensed under the MIT License


## References :book:
- Steffen Rendle, Lars Schmidt-Thieme. Online-updating regularized kernel matrix factorization models for large-scale recommender systems https://dl.acm.org/doi/10.1145/1454008.1454047
