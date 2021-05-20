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

* Code 

  * ```python
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
    
    # 1. train a model with all features
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=2, random_state=2909, n_jobs=4)
    
    rf.fit(X_train, y_train)
    
    # 2. shuffle features and assess performance drop
    train_roc = roc_auc_score(y_train, (rf.predict_proba(X_train))[:, 1])
    
    # list to capture the performance shift
    performance_shift = []
    
    # selection  logic
    for feature in X_train.columns:
    
        X_train_c = X_train.copy()
    
        # shuffle individual feature
        X_train_c[feature] = X_train_c[feature].sample(
            frac=1, random_state=10).reset_index(drop=True)
    
        # make prediction with shuffled feature and calculate roc-auc
        shuff_roc = roc_auc_score(y_train, rf.predict_proba(X_train_c)[:, 1])
        
        drift = train_roc - shuff_roc
    
        # save the drop in roc-auc
        performance_shift.append(drift)
        
    
    # 3. capture the selected features
    feature_importance = pd.Series(performance_shift)
    feature_importance.index = X_train.columns
    selected_features = feature_importance[feature_importance > 0].index
    
    ```



## Recursive feature elimination

* Process 
  * Build a model and calculate feature importance (model 1)
  * Remove the least important feature 
  * Rebuild a model (model 2)
  * Reevaluate the performance 
  * If drop in performance (model 1 - model 2) is bigger than the threshold, it is an important feature, so keep it. If not, remove the feature. 

* Code

  * ```python
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import roc_auc_score, r2_score
    
    # 1. Build ML model with all features
    model_full = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=10)
    
    model_full.fit(X_train, y_train)
    
    y_pred_test = model_full.predict_proba(X_test)[:, 1]
    roc_full = roc_auc_score(y_test, y_pred_test)
    
    # 2. Rank features by importance
    features = pd.Series(model_full.feature_importances_)
    features.index = X_train.columns
    features.sort_values(ascending=True, inplace=True)
    features = list(features.index)
    
    # 3. Select features
    # recursive feature elimination:
    
    # first we arbitrarily set the drop in roc-auc
    # if the drop is below this threshold,
    # the feature will be removed
    tol = 0.0005
    
    print('doing recursive feature elimination')
    
    # we initialise a list where we will collect the
    # features we should remove
    features_to_remove = []
    
    # set a counter to know where the loop is
    count = 1
    
    # now we loop over all the features, in order of importance:
    # remember that features is this list are ordered
    # by importance
    for feature in features:
        
        print()
        print('testing feature: ', feature, count, ' out of ', len(features))
        count = count + 1
    
        # initialise model
        model_int = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=10)
    
        # fit model with all variables, minus the feature to be evaluated
        # and also minus all features that were deemed to be removed
        
        # note that features_to_remove will be empty in the first rounds
        # but will have features as the loop proceeds
        model_int.fit(
            X_train.drop(features_to_remove + [feature], axis=1), y_train)
    
        # make a prediction using the test set
        y_pred_test = model_int.predict_proba(
            X_test.drop(features_to_remove + [feature], axis=1))[:, 1]
    
        # calculate the new roc-auc
        roc_int = roc_auc_score(y_test, y_pred_test)
        print('New Test ROC AUC={}'.format((roc_int)))
    
        # print the original roc-auc with all the features
        print('Full dataset ROC AUC={}'.format((roc_full)))
    
        # determine the drop in the roc-auc
        diff_roc = roc_full - roc_int
    
        # compare the drop in roc-auc with the tolerance
        # we set previously
        if diff_roc >= tol:
            print('Drop in ROC AUC={}'.format(diff_roc))
            print('keep: ', feature)
            print
        else:
            print('Drop in ROC AUC={}'.format(diff_roc))
            print('remove: ', feature)
            print
            # if the drop in the roc is small and we remove the
            # feature, we need to set the new roc to the one based on
            # the remaining features
            roc_full = roc_int
            
            # and append the feature to remove to the collecting list
            features_to_remove.append(feature)
    
    # now the loop is finished, we evaluated all the features
    print('DONE!!')
    print('total features to remove: ', len(features_to_remove))
    
    # determine the features to keep (those we won't remove)
    features_to_keep = [x for x in features if x not in features_to_remove]
    print('total features to keep: ', len(features_to_keep))
    
    ```



## Recursive feature addition 

* Process 
  * Build a model with all features and calculate feature importance
  * Pick the most important feature and build a model (model 1)
  * Calculate the initial performance 
  * Add the second important feature and build a model (model 2)
  * If performance increase is bigger than threshold, it is an important feature, so keep it. If not, remove the feature. 
  * Repeat until all the features in the dataset is examined 

* Code

  * ```python
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import roc_auc_score, r2_score
    
    # 1. Train a ML model with all features
    # build initial model using all the features
    model_full = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=10)
    
    model_full.fit(X_train, y_train)
    
    # calculate the roc-auc in the test set
    y_pred_test = model_full.predict_proba(X_test)[:, 1]
    roc_full = roc_auc_score(y_test, y_pred_test)
    
    
    # 2. Get feature importance 
    # get feature name and importance
    features = pd.Series(model_full.feature_importances_)
    features.index = X_train.columns
    
    # sort the features by importance
    features.sort_values(ascending=False, inplace=True)
    
    # 3. Build a model with 1 features 
    # build initial model using all the features
    model_one_feature = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=10)
    
    # train using only the most important feature
    model_one_feature.fit(X_train[features[0]].to_frame(), y_train)
    
    # calculate the roc-auc in the test set
    y_pred_test = model_one_feature.predict_proba(X_test[features[0]].to_frame())[:, 1]
    
    roc_first = roc_auc_score(y_test, y_pred_test)
    
    print('Test one feature xgb ROC AUC=%f' % (roc_first))
    
    
    # 4. select features
    # recursive feature addition:
    
    # first we arbitrarily set the increase in roc-auc
    # if the increase is above this threshold,
    # the feature will be kept
    tol = 0.0001
    
    print('doing recursive feature addition')
    
    # we initialise a list where we will collect the
    # features we should keep
    features_to_keep = [features[0]]
    
    # set a counter to know which feature is being evaluated
    count = 1
    
    # now we loop over all the features, in order of importance:
    # remember that features in the list are ordered
    # by importance
    for feature in features[1:]:
        print()
        print('testing feature: ', feature, count, ' out of ', len(features))
        count = count + 1
    
        # initialise model
        model_int = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=10)
    
        # fit model with the selected features
        # and the feature to be evaluated
        model_int.fit(
            X_train[features_to_keep + [feature] ], y_train)
    
        # make a prediction over the test set
        y_pred_test = model_int.predict_proba(
            X_test[features_to_keep + [feature] ])[:, 1]
    
        # calculate the new roc-auc
        roc_int = roc_auc_score(y_test, y_pred_test)
        print('New Test ROC AUC={}'.format((roc_int)))
    
        # print the original roc-auc with one feature
        print('Previous round Test ROC AUC={}'.format((roc_first)))
    
        # determine the increase in the roc-auc
        diff_roc = roc_int - roc_first
    
        # compare the increase in roc-auc with the tolerance
        # we set previously
        if diff_roc >= tol:
            print('Increase in ROC AUC={}'.format(diff_roc))
            print('keep: ', feature)
            print
            # if the increase in the roc is bigger than the threshold
            # we keep the feature and re-adjust the roc-auc to the new value
            # considering the added feature
            roc_first = roc_int
            
            # and we append the feature to keep to the list
            features_to_keep.append(feature)
        else:
            # we ignore the feature
            print('Increase in ROC AUC={}'.format(diff_roc))
            print('remove: ', feature)
            print
    
    # now the loop is finished, we evaluated all the features
    print('DONE!!')
    print('total features to keep: ', len(features_to_keep))
    ```

  * 