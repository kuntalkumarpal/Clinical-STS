# Clinical-STS
Clinical Semantic Textual Similarity


## Methods 

### SML
* AdaBoost
* BayesianRidgeRegression
* DecisionTreeClassifier
* ensemble
* Extra Trees
* Gradient Boost
* Lasso Regression
* LassoLarsRegression
* LinearRegression
* LogisticRegression
NeuralNetSingle
* RandomForestRegression
* SVM_OneVsAllClassifier
* SVM_OneVsOneClassifier
* XGBoostClassifier
* XGBoostRegressor

### DL
* Deep Neural Network 1 layer
* Deep Neural Network Multi layer BioSentVec
* Deep Neural Network Multi layer
* Deep Neural Network ReLU

### Features
* Biomedical sentence embedding, BioSentVec
  * Cosine distance, Euclidean distance, Squared-Euclidean Distance, Correlation and Word-Mover distance 
* Token-level similarity
  * Jaccard (threshold of 0.7), Q-gram(q=2,3,4), Cosine, Dice, Overlap-based, Tversky Index, Monge-Elkan, Affine, Bag-Distance, TF-IDF, Editex, Levenstein, Needleman-Wunsh and Smith-Waterman similarity both for the given sentence pairs and also for the modified sentences having a common prefix. 
* Numerical similarity we converted them into words and evaluate through a 200-dimension BioWordVec5 model.
* Natural language inference-based(NLI) features for the task. 
* Clinical concepts similarity using Metamap
