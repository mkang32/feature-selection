# Intro

## Filter methods

* Characteristics 

  * Rely on the charateristics of the data (feature characteristics)
  * Do not use machine learning algorithms 
  * Model agnostic
  * Tend to be less computationally expensive 
  * Usually give lower prediction performance than a wrapper methods 
  * Are very well suited for a quick screen and removal of irrelavant features 

* Procedure (univariate)

  * Rank features according to a certain criteria 
    * Each feature is ranked independently of the feature space 
  * Select the highest ranking features

  * May select redundant variables because they do not consider the relationships between features 

* Ranking criteria 

  * Chi-square | Fisher score 
  * Univariate parametric tests (ANOVA) 
  * Mutual Information 
  * Variance 
    * Constant features 
    * Quasi-constant features 

* Multivariate 
  * Handle redundant feature 
  * Duplicated features 
  * Correlated features 



## Wrapper methods

* Characteristics 
  * Use predictive machine learning models to score the feature subset 
  * Train a new model on each feature subset 
  * Tend to be very computationally expensive 
  * Usually provide the best performing feature subset for a given machine learning algorithm 
  * Find the optimal feature subset for the desired classfier but they may not produce the best feature combination for a different machine learning model 
* Procedure 
  * Search for a subset of features 
  * build a machine learning model on the selected feature subset 
  * Evaluate model performance 
  * Repeat until criteria is met 
* Search mechanisms
  * Forward selection
    * Adds 1 feature at a time until predefined criteria is met 
  * Backward selection
    * Starts with all the features and removes 1 feature at a time 
  * Exhaustive search 
    * Searches across all possible feature combinations 

* Search algorithm 
  * Greedy algorithms 
  * Aim to find the best possible combinations 
  * Computationally expensive 
  * Often impracticable (exhaustive search)
* Stopping criteria 
  * Performance does not increase (forward selection)
  * Performance does not decrease (backwards elimination)
  * Predefined number of features is reached 
  * These need to be defined by user 



## Embedded methods 

* Characteristics 
  * Perform feature selection as part of the model construction process 
  * Consider the interaction between features and models 
  * They are less computationally expensive than wrapper methods, because they fit the model learning model only once 
* Pros 
  * Faster than wrapper methods 
  * More accurate than filter methods 
  * Detect interactions between variables 
  * Find the feature subset for the algorithm being trained 
* Procedure 
  * Train a machine learning algorithm 
  * Derive the feature imporatnace 
  * Remove non-important features 
* Examples 
  * LASSO
  * Tree importance 
  * Regression coefficients 



# Filter Methods

## Filter Basics

## Filter Correlation

## Filter Statistical Tests

## Filter Other Metrics

### Univariate Model Performance Metrics

* Process 
  * Build a model with one feature 
  * Measure performace metrics (e.g. ROC-AUC)
  * Repeat for all features 
  * Rank the features and select the top rank features 

* Pros 
  * We can use any machine learning algorithm 
  * We can use any performance metric (e.g. ROC-AUC, accuracy, precision, recall, MSE, RMSE, etc.)

* Caveat 
  
* Feature subsets may depend on machine learning algorithm used (not model diagnostic) and metric used 
  
* Code 

  ```python
  # classification
  roc_values = []
  
  for feature in X_train.columns: 
      clf = DecisionTreeClassifier()
      clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
      
      # obtain the predictions
      y_scored = clf.predict_proba(X_test[feature].to_frame())
      
      # calculate the score 
      roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
   
  # select features
  roc_values = pd.Series(roc_values)
  roc_values.index = X_test.columns
  selected_features = roc_values[roc_values > 0.5].index
  ```

  



# Wrapper Methods

## Step Foward 

* Process 
  * Build models with one feature at a time to find the most predictive feature (e.g. F2)
  * Build models with two features including F2 that are most predictive (e.g. F2 + F1)
  * Build models with three features including F2 + F1 that are most predictive (e.g. F2 + F1 + F4)
  * Repeat until performance does not increase beyond a threshold defined by the user



## Step Backward 



## Exhaustive Search





# Embedded Methods



# Hybrid Feature Selection

