# What is PCA?
Principal component analysis (PCA) is a technique that transforms high-dimensions data into lower-dimensions while retaining as much information as possible.

![image](https://user-images.githubusercontent.com/86812576/171867562-b2561b02-6391-45ee-8a09-6a5709527c75.png)

The original 3-dimensional data set. The red, blue, green arrows are the direction of the first, second, and third principal components, respectively.

![image](https://user-images.githubusercontent.com/86812576/171867644-8b741ede-3b78-405c-902d-f54b4ccc4af0.png)

Scatterplot after PCA reduced from 3-dimensions to 2-dimensions.

PCA is extremely useful when working with data sets that have a lot of features. Common applications such as image processing, genome research always have to deal with thousands-, if not tens of thousands of columns.

While having more data is always great, sometimes they have so much information in them, we would have impossibly long model training time and the curse of dimensionality starts to become a problem. Sometimes, less is more (source: https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d).

# Breast-Cancer-Classification-with-PCA-using-PyTorch
# Import Packages
import common package:

- import **numpy as np**
- import **pandas as pd**
- import **matplotlib.pyplot as plt**
- import **seaborn as sns**
- from **sklearn.model_selection** import **train_test_split**
- from **sklearn.pipeline** import **Pipeline**
- from **sklearn.compose** import **ColumnTransformer**
- from **jcopml.utils** import **ave_model, load_model**
- from **jcopml.pipeline** import **num_pipe, cat_pipe**
- from **jcopml.plot** import **plot_missing_value**
- from **jcopml.feature_importance** import **mean_score_decrease**

import algorithm's package:
- from **sklearn.decomposition** import **PCA**
- from **sklearn.svm** import **SVC**
- from **sklearn.model_selection** import **RandomizedSearchCV**
- from **jcopml.tuning** import **random_search_params as rsp**
- from **jcopml.tuning.space** import **Real, Integer**

# Import Data
The data consists of 569 rows and 31 columns. If we look at the data, the many columns can actually be reduced and grouped into several columns. Because some of the columns have similarities with each other. And that's what PCA will do, which is to reduce the column to fewer but still rich in information.

# Dataset Splitting
Split data into X, and y.

X = all columns except the target column.

y = 'target' column as target

test_size = 0.2 (which means 80% for train, and 20% for test). And stratified so that the test data is representative.

# Visualize how PCA can help
On PCA, I will make the feature only 2. And add whiten, which is standard scaling because behind the scenes, PCA uses gradient descent.

![image](https://user-images.githubusercontent.com/86812576/171877747-045b55d2-fefd-43e2-9f8f-d755e997e303.png)

It can be seen that the data in two dimensions is separated after PCA is performed.

# Training
In the Training step there are 3 main things that I specify.

First, the preprocessor: here the columns will be grouped into numeric and categoric.

But we don't have categorical data, so it will only create numeric columns.

second, pipeline: contains the preprocessor as 'prep' which I defined earlier, and the algorithm as 'algo' which in this case I use Support Vector Classifer. as well as inserting the 'PCA' algorithm into the pipeline.

and third, tuning with RandomizedSearchCV: in this case I use the tuning recommendations (rsp.xgb_params) that often occur in many cases. but does not rule out hyperparameter tuning if the model results are not good. As well as added tuning for PCA parameters, added PCA components, and whiten. with cross validation = 3, and n_iter = 50 (trials in Random Search)

# Result
![Screenshot 2022-06-03 220326](https://user-images.githubusercontent.com/86812576/171880687-fd13f2cf-a828-4bab-9dfc-03dfe29087b3.png)

The result can be seen that the machine algorithm chooses 19 PCA components from the initial 30 features. meaning that these 19 features are important features and can already represent the previous 30 features. We don't know what the 19 features are, but they are the result of a linear combination of the previous 30 features. the score is also high, because during the previous visualization it was seen that the data had been separated.

# How to determine n_components -> Cumulative Explained Variance
Previously we used tuning to determine the components of the PCA. It turns out that there is actually a technique to determine the number of components, how many components must be reduced so that the information is maintained?

![image](https://user-images.githubusercontent.com/86812576/171884814-c5bdf7d3-81ed-4e7b-b68f-0edd1e07add6.png)

it can be seen in n components 30, the cumulative explained variance is 100%. This means that the information is still 100% preserved. When the component (feature) is reduced, the cumulative explained variance will continue to decrease. But because of the data in this case, the features may be correlated with each other, the cumulative variance explained when the reduced features are not very visible, and only decreases below 1%. So when it is compressed to 5 components, the information is only lost a little.

# Training after determine n_components
Because we have done the component determination manually with cumulative explained variance, then I will do training once again with 5 components.

![Screenshot 2022-06-03 223137](https://user-images.githubusercontent.com/86812576/171887269-8ac777d3-31d3-4a2f-83fa-53616fc1368e.png)

 Sure enough, the score had indeed decreased by less than 1%. In the first training with 19 features, the actual information is relatively the same.
