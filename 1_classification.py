#./bin/spark-submit 1_classification.py

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Practise').getOrCreate()

data = spark.read.csv("1_classification.csv", inferSchema = True, header=True)
#data.schema.names # print list of columns


#################
## Format data ##
#################

################################################################
# REMEMBER! Spark does not like column names containing "."  ###
################################################################

#Rrturns a new DataFrame by renaming an existing column.
data = data.withColumnRenamed('F.Undergrad', 'F_Undergrad')
data = data.withColumnRenamed('P.Undergrad', 'P_Undergrad')
data = data.withColumnRenamed('Room.Board', 'Room_Board')
data = data.withColumnRenamed('S.F.Ratio', 'SF_ratio')
data = data.withColumnRenamed('perc.alumni', 'perc_alum')
data = data.withColumnRenamed('Grad.Rate', 'grad_rate')

################################################################
#### Vector Assembler  									     ###
################################################################
#data.printSchema()
from pyspark.ml.feature import VectorAssembler
# inputCols= list to take all columns that are to be features in ML classification algorithm
# outputCols = name of output column eg. "features"
assembler = VectorAssembler(inputCols=['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F_Undergrad', 'P_Undergrad', 'Outstate', 'Room_Board', 'Books', 'Personal', 'PhD', 'Terminal', 'SF_ratio', 'perc_alum', 'Expend', 'grad_rate'],outputCol='features')  
output = assembler.transform(data)


################################################################
#### Change String labels to Numeric Inexes					 ###
################################################################

# we wish to predict the Private column
# we need to change string "yes" or "no" to 1 or 0
from pyspark.ml.feature import StringIndexer 
indexer = StringIndexer(inputCol='Private', outputCol='PrivateIndex')
output_fixed = indexer.fit(output).transform(output)
#output_fixed.printSchema()
final_data = output_fixed.select('features','PrivateIndex') # features + y 

################################################################
#### STANDARD TREE METHOD PIPELINE 							 ###
################################################################
train_data, test_data = final_data.randomSplit([0.7, 0.3], seed=666)
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, RandomForestClassifier

#from pyspark.ml import PipeLine 

dtc = DecisionTreeClassifier(labelCol='PrivateIndex', featuresCol = 'features')
rfc = RandomForestClassifier(labelCol='PrivateIndex', featuresCol = 'features')
gbt = GBTClassifier(labelCol='PrivateIndex', featuresCol = 'features')

#dtc_model = dtc.fit(train_data)
#rfc_model = rfc.fit(train_data)
#gbt_model = gbt.fit(train_data)

#dtc_preds = dtc_model.transform(test_data)
#rfc_preds = rfc_model.transform(test_data)
#gbt_preds = gbt_model.transform(test_data) #maxDepth=5)
 

#from pyspark.ml.evaluation import BinaryClassificationEvaluator

#my_binary_eval =  BinaryClassificationEvaluator(labelCol="PrivateIndex") #BinaryClassificationEvaluator() 
#print("DTC")
#print(my_binary_eval(dtc_preds))
#print("Random Forrest results:")
#print(my_binary_eval(rfc_preds))
#my_binary_eval2 =  BinaryClassificationEvaluator(labelCol="PrivateIndex", rawPredictionCol='prediction') #BinaryClassificationEvaluator() 
#print("Gradient Boosting results:")
#print(my_binary_eval2(gbt_preds))


#from pyspark.ml.evaluation import MulticlassClassificationEvaluator 
#acc_eval = MulticlassClassificationEvaluator(labelCol = 'PrivateIndex', metricName='accuracy')
## other metrics "weightedPrecision", "weightedRecall, 'f1'
#rfc_acc = acc_eval.evaluate(rfc_preds)
#print(rfc_acc)


######################################
#####   CLUSTERING           #########
######################################

# remove labels column
# Scale everything

from pyspark.ml.feature import StandardScaler
#final_data.printSchema()
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
scaler_model = scaler.fit(final_data)
final_data = scaler_model.transform(final_data)

from pyspark.ml.clustering import KMeans
# apply k-means clustering
kmeans = KMeans(featuresCol='scaledFeatures', k=3)
model = kmeans.fit(final_data)

print("WSSE:")
print(model.computeCost(final_data))

centers = model.clusterCenters()
print("centers:'")
print(centers)

model.transform(final_data).select('prediction').show()
















