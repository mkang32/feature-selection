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

* Process 
  * e.g. There are 4 features (F1, F2, F3, F4)
  * Build models with all 4 combiation of 3 features (e.g. F1 + F2 + F3 vs. F1 + F2 + F4 vs. F1 + F3 + F4 vs. F2 + F3 + F4 ) and find the most predictive combination 
  * Build  models with 2 features 
  * Repeat until performance does not decrease beyond a threshold 



## Exhaustive Search

* Process 

  * Try all possible feature cominations and find the best performing combination
  * In practice, define the minimum and maximum number of features of the subsets to test 

* Code 

  ```python
  from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
  from sklearn.ensemble import RandomForestRegressor
  
  efs = EFS(RandomForestRegressor(n_estimators=5,
                                  n_jobs=4,
                                  random_state=0,
                                  max_depth=2),
            min_features=1,
            max_features=2,
            scoring='r2',
            print_progress=True,
            cv=2)
  
  efs = efs.fit(np.array(X_train), y_train)
  
  # selected columns
  selected_feat = X_train.columns[list(efs.best_idx_)]
  selected_feat
  ```

  



# Embedded Methods

## Regression Coefficients

* The coefficients of the predictors are directly proportional to how much that feature contributes to the final value of y 
* Under the condition of the following assumptions:
  * Linear relationship between predictor (X) and outcome (Y)
  * Xs are independent 
  * Xs are not correlated to each other (no-multicollinearity)
  * Xs are normally distributed 
  * For direct coefficient comparison Xs should be in the same scale 

* Code

  ```python
  from sklearn.preprocessing import StandardScaler 
  from sklearn.feature_selection import SelectFromModel
  from sklearn.linear_model import LogisticRegression
  
  # scale 
  scaler = StandardScaler()
  scaler.fit(X_train)
  
  # train 
  sel_ = SelectFromModel(
      LogisticRegression(C=1000, penalty='l2', max_iter=300, random_state=10)
  )
  
  sel_.fit(scaler.transform(X_train), y_train)
  
  # selected features 
  selected_feat = X_train.columns[sel_.get_support()]
  
  ```

  



## Lasso regularization 

* Regularization 

  * Adds a penalty on the parameters of the model to reduce the freedom of the model. 
  * Less overfit, better generalization. 
  * For linear models, there are three types of regularization: 
    * L1 (Lasso)
    * L2 (Ridge)
    * L1/L2 (Elastic net)

* L1 (Lasso)

  * $$
    \frac{1}{2m}\sum(y-\hat{y})^2 + \lambda \sum \phi
    $$

  * $\hat{y}=\theta_1 X_1 + \theta_2 X_2 + ... + \theta_n X_n$

  * $\lambda$ is the regularization parameter = penalty. Higher the penalty, the bigger the generalization. If the penalty is too high, the model may lose predictive power. 

  * L1 will shrink some parameters to zero, allowing for feature elimination 

* L2 (Ridge)

  * $$
    \frac{1}{2m}\sum(y-\hat{y})^2 + \lambda \sum \phi^2
    $$

  * $\hat{y}=\theta_1 X_1 + \theta_2 X_2 + ... + \theta_n X_n$

  * $\lambda$ is the regularization parameter = penalty. Higher the penalty, the bigger the generalization. If the penalty is too high, the model may lose predictive power. 

  * L2 will make coefficients approach to zero but equal to zero. No variable is ever excluded.  



## Trees

* Decision Tree feature importance 
  * How much a feature decrease impurity? = How good the feature is at separating the classes? 
  * Features on the higher nodes have greater gains in impurity, meaning more important ones. 
  * Meausre of impurity 
    * Classficiation => Gini or entropy
    * Regression => variance 
* Feature importance in random forest 
  * Average of the feature importance across tress 
  
  * Note: 
    * RF in general give preferenec to features with high cardinality 
    * Correlated features will have the same or similar importance, but reduced importance compared to the same tree built without correlated counterparts 
    
  * Code
  
    * ```python
      
      ```
  
    * 
* Recursive feature elimination using random forest 
  * Build a random forest model 

  * Calculate feature importance 

  * Remove the least important feature 

  * Repeat until a condition is met 
    
    * Usually the number of elimination or magnitude of smallest importance
    
  * Pros 
    
    * Eliminate one of the highly correlated features --> the other correlated feature will have high feature importance after eliminating one 
    
  * Cons

    * Computationally expensive 

  * Code

    * ```python
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.feature_selection import RFE # recursive feature elimination
      
      sel_ = RFE(RandomForestClassifier(n_estimators=10, random_state=10), n_features_to_select=27)
      sel_.fit(X_train, y_train)
      
      selected_feat = X_train.columns[sel_.get_support()]
      ```

  



# Hybrid Feature Selection

## Feature shuffling

* Process
  * Shuffle values in one feature, evaluate the performance drop 
  * Repeat the first step for other features 
  * Compare the performance drop of different features and select the ones with performance drop above threshold (this indicates importance of the feature)

