# TIM-209 Warm-Up Project

## 1. Iris_data(classification)
http://archive.ics.uci.edu/ml/datasets/Iris  
The classic classification dataset, contains 150 instances. Attributes for each instances  
1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm  
5. **classification label:**  
-- Iris Setosa(0)  -- Iris Versicolour(1)  -- Iris Virginica(2)

Methods  used:  KNN, SVM and DecisionTree  
Cross-Validation: k=5  
Measurement score:  # of correct classification / total number  
KNN= 0.953  
SVM = 0.973  
DT = 0.953  

## 2. Wine_data(classification)
http://archive.ics.uci.edu/ml/datasets/Wine+Quality  
Using chemical analysis determine the origin of wines, contains 178 instances. Attributes for each instance  
1. fixed acidity   
2. volatile acidity   
3. citric acid  
4. residual sugar   
5. chlorides   
6. free sulfur dioxide   
7. total sulfur dioxide   
8. density   
9. pH   
10. sulphates   
11. alcohol   
12. **classification label:**  
quality (score between 0 and 10) 
 
Methods  used:  KNN, SVM and DecisionTree  
Cross-Validation: k=10  
Measurement score:  # of correct classification / total number  
KNN = 0.711  
SVM = 0.447  
DT = 0.894

## 3. Breast_Cancer_Coimbra data (classification)
http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra  
Clinical features were observed or measured for 64 patients with breast cancer and 52 healthy controls, 
which in total 114 instances. Attributes for each instance  
1. Age (years) 
2. BMI (kg/m2) 
3. Glucose (mg/dL) 
4. Insulin (µU/mL) 
5. HOMA 
6. Leptin (ng/mL) 
7. Adiponectin (µg/mL) 
8. Resistin (ng/mL) 
9. MCP-1(pg/dL)   
10. **classification label:**   
1=Healthy controls 2=Patients

Methods  used:  KNN, SVM and DecisionTree  
Cross-Validation: k=5  
Measurement score:  # of correct classification / total number  
KNN = 0.513  
SVM = 0.547  
DT = 0.591

## 4. diabetes_data(prediction)
https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html  
This is scikit-learn linear regression test dataset, which contains 442 instances. Attributes for each instance  
1. age 
2. sex 
3. body mass index
4. average blood pressure  
5. six blood serum measurements(counted as 6 attributes) 
 
**Prediction output : Y-quantitative measure of disease progression** 
 
Methods used: Linear(Ridge) Regression, Lasso and Bayesian Regression  
Cross-Validation: k=5  
Measurement score: mean-square-error of prediction and true value  
LR = 2994.28  
Lasso = 3851.22  
Bayesian = 2986.69  



## 5. auto_mpg data(prediction)
http://archive.ics.uci.edu/ml/datasets/Auto+MPG  
The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes, which contains 398 instances, what we need to predict is mpg value. Attributes for each instance  
2. cylinders: multi-valued discrete 
3. displacement: continuous 
4. horsepower: continuous 
5. weight: continuous 
6. acceleration: continuous 
7. model year: multi-valued discrete 
8. origin: multi-valued discrete 
9. car name: string (unique for each instance)  

**Prediction output: mpg value**  

Methods used: Linear(Ridge) Regression, Lasso and Bayesian Regression  
Cross-Validation: k=5  
Measurement score: mean-square-error of prediction and true value    
LR = 11.41  
Lasso = 12.07  
Bayesian = 11.43    

## 6. airfoil self-noise data(prediction)
http://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise  
NASA data set, obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic wind tunnel, which contains 1503 instances. Attributes for each instance
1. Frequency, in Hertzs  
2. Angle of attack, in degrees   
3. Chord length, in meters  
4. Free-stream velocity, in meters per second   
5. Suction side displacement thickness, in meters
 
**Prediction output:  Scaled sound pressure level** 
 
Methods used: Linear(Ridge) Regression, Lasso and Bayesian Regression  
Cross-Validation: k=5  
Measurement score: mean-square-error of prediction and true value  
LR = 23.23 
Lasso = 35.06  
Bayesian = 23.24  



