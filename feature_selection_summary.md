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
  * Repeat 
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



# Wrapper Methods



# Embedded Methods



# Hybrid Feature Selection

