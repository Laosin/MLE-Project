from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Split into training and inference sets
train_set, inference_set = train_test_split(iris_df, test_size=0.2, random_state=42)

# Save to CSV files
train_set.to_csv('datafiles/train_set.csv', index=False)
inference_set.to_csv('datafiles/inference_set.csv', index=False)

