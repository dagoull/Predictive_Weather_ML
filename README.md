# Towards a Predictive Weather Model Using Machine Learning In Tenerife (Canary Islands)
This repo contains a set of R scripts for preprocessing, transformation, training, optimization and validation. We used the caret package (Classification and Regression Training) (Khun, 2008). Caret is an interface that unifies under just one framework several machine learning packages, making data preprocessing, training, optimization and validation of predictive models easier and with native support for parallel calculations (Khun & Johnson, 2013).
#### Setup
```
R, version 3.4 or later. Package: caret, version: 6.0-80, package caretEnsemble, version 2.0.0
```
#### Main algorithms
```
- Logistic Model Trees (lmt)
- Linear Discriminant Analysis (lda)
- Generalized Linear Model (glm)
- Support Vector Machines (svmPoly)
- Random Forest (rf)
- Stochastic Gradient Boosting (gbm) 
- eXtreme Gradient Boosting (XGBoost)
```
#### Dataset
```
The dataset used in this paper is available for download from [ https://data.mendeley.com/datasets/srwzh55hrz/1 ] (https://data.mendeley.com/datasets/srwzh55hrz/1)
```
[ https://data.mendeley.com/datasets/srwzh55hrz/1 ] (https://data.mendeley.com/datasets/srwzh55hrz/1)

## References
* Kuhn, M. (2008). Caret package. Journal of Statistical Software, 28(5), 1-26. 
* Kuhn, M., & Johnson, K. (2013). Applied predictive modeling (Vol. 26). New York: Springer.
* Kuhn, M., & Johnson, K. (2013). A Short Tour of the Predictive Modeling Process. In Applied predictive modeling (pp. 19-26). Springer, New York, NY.
