#./bin/spark-submit 1_regression.py
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('LinearRegressionPractise').getOrCreate()

from pyspark.ml.regression import LinearRegression

data = spark.read.csv("1_regression.csv", inferSchema = True, header=True)
#data.printSchema() # check datatype of each column

##############################################
## Check for correlation between features ####
##############################################
from pyspark.sql.functions import corr 
data.select(corr('Avg Session Length', 'Time on Website')).show()
data.select(corr('Avg Session Length', 'Time on App')).show()
data.select(corr('Time on App', 'Time on Website')).show()

# DATA WRANGLING PACKAGES NEEDED FOR MACHINE LEARNING 
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


#####################################################################
## For any machine learning algorithm							#####
## Spark needs a particular format								#####
## Where labels are in one column								#####
## And a features column - containing all selected features		#####
#####################################################################

#assembler = VectorAssembler(inputCols=['future_dif', 'held_weight', 'shell_weight'], outputCol='features')
assembler = VectorAssembler(inputCols=['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership'], outputCol='features')



output = assembler.transform(data) # adds a "features" column 
# dependent variable = "Yearly Amount Spent"
final_data = output.select('features', 'Yearly Amount Spent') 

### Train vs Test split
train_data, test_data = final_data.randomSplit([0.7, 0.3])
#train_data.describe().show() #summary statistics of label column 

###################################
## Fit Linear Regression Model  ###
###################################
# train model on training set
lr = LinearRegression(labelCol = 'Yearly Amount Spent')
lr_model = lr.fit(train_data)
# predict test set
test_results = lr_model.evaluate(test_data)


###################################
## EVALUATE LINEAR REGRESSION  ####
###################################

#test_results.residuals.show() # residuals
print(test_results.rootMeanSquaredError)  # root mean squared error
print(test_results.r2) # R squared
print(test_results.meanAbsoluteError)
print(test_results.meanSquaredError)


###########################################################
#### LINEAR REGRESSION WITH STOCHASITC GRADIENT DESCENT ###
####  + Practise using RDDs instead of Spark Dataframe  ###
###########################################################
# http://www.techpoweredmath.com/spark-dataframes-mllib-tutorial/#.WdAHi2Usip5

# convert dataframe to RDD
my_rdd = final_data.rdd


rdd_labels = my_rdd.map(lambda row: row[1])
rdd_features = my_rdd.map(lambda row: row[0])

#Now the two RDDs can be put together with 'zip'
transformedData = rdd_labels.zip(rdd_features)
#print(transformedData.take(5))

# go back to the LabeledPoint structure before using MLlib.
from pyspark.mllib.regression import LabeledPoint
transformedData = transformedData.map(lambda row: LabeledPoint(row[0],[row[1]]))
#print(transformedData.take(5))
trainingData, testingData = transformedData.randomSplit([.8,.2],seed=1234)


from pyspark.mllib.regression import LinearRegressionWithSGD
lr_SGD = LinearRegressionWithSGD.train(trainingData,iterations=100,step=1.0, miniBatchFraction=1.0)
print(lr_SGD.weights)
#print(lr_SGD.take(10))

test_features = testingData.map(lambda row: row[0])
predictions = lr_SGD.predict(test_features)
print(predictions)

#####################
## EVALUATION #######
#####################
from pyspark.mllib.evaluation import RegressionMetrics


#We need an RDD that's a tuple of predictions from our model and the original home values. 
# RESUBSTITUTION ERROR
prediObserRDDin = trainingData.map(lambda row: (float(lr_SGD.predict(row.features[0])),row.label))
metrics = RegressionMetrics(prediObserRDDin)
print(metrics.r2)

# GENERALIZATION ERROR
#prediObserRDDout = testingData.map(lambda row: (float(lr_SGD.predict.predict(row.features[0])),row.label))
#metrics_2 = RegressionMetrics(prediObserRDDout)
#print(metrics_2.rootMeanSquaredError)


#Other regression functions:
# DecisionTreeRegressor(), GBTRegressor() and RandomForestRegressor()
#GeneralizedLinearRegression() - supports different error distributions
#								# gaussian, binomial, gamma and poisson.


###################################################
###     LOGISTIC REGRESSION CLASSIFICTION        ##
###################################################


#https://spark.apache.org/docs/2.0.2/ml-features.html
#https://books.google.co.uk/books?id=HVQoDwAAQBAJ&pg=PA98&lpg=PA98&dq=pyspark+continuous+variables+into+binary&source=bl&ots=tLFqNhEgbH&sig=UiT8nAfOB6uzvNLwMfTEGAw9dk0&hl=en&sa=X&ved=0ahUKEwjTqrP4-c3WAhWOZFAKHdN6CKEQ6AEINDAC#v=onepage&q=pyspark%20continuous%20variables%20into%20binary&f=false

# Useful transformation functions:
#binarizer: converts continuous variables to 1 / 0 depending on set threshold 
#bucketizer: ^ similar but for multi-class problems
#MaxAbsScaler: rescale data between -1 and 1 range
#MinMaxSacler: rescale data between 0 and 1 range
#OneHotEncoder: encodes categorical column to binary vectors
#PCA: self explanatory
#StandardScaler: convert so mean = 0 and sd = 1 


from pyspark.ml.feature import Binarizer
binarizer = Binarizer(threshold=500, inputCol="Yearly Amount Spent", outputCol="label")
binarizedDataFrame = binarizer.transform(final_data)
binarizedDataFrame = binarizedDataFrame.drop("Yearly Amount Spent")
binarizedDataFrame.show()

from pyspark.ml.classification import LogisticRegression
logReg = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
fitted_logReg = logReg.fit(binarizedDataFrame)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(fitted_logReg.coefficients))
print("Intercept: " + str(fitted_logReg.intercept))
#log_summary = fitted_logReg.summary()

# test set
predictions_and_labels = fitted_logReg.evaluate(fitted_logReg) 

#from pyspark.ml.evaluation import BinaryClassificationEvaluator
#my_eval = BinaryClassificationEvaluator()
#roc_results = my_eval.evaluate(predictions_and_labels.predictions)


#
from spark.ml.regression import RandomForestRegressor

