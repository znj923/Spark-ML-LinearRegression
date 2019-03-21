from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder.appName('LRExample02').getOrCreate()

data = spark.read.csv('file:///D:/Spark/spark-2.3.3-bin-hadoop2.7/data/Linear_regression_house-master/boston.csv',header=True, inferSchema=True)
data.show(10)

# 合并特征
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
                                             'ptratio', 'black', 'lstat'], outputCol='features')
v_data = vectorAssembler.transform(data)
v_data.show(10)

# 划分训练集，集测试集
vdata = v_data.select(['features', 'medv'])
vdata.show(10)
splits = vdata.randomSplit([0.7, 0.3])
train_data = splits[0]
test_data = splits[1]

# 训练
lr = LinearRegression(featuresCol='features', labelCol='medv', maxIter=10000, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(train_data)

print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# 预测
predictions = lrModel.transform(test_data)
predictions.select('features', 'medv', 'prediction').show()

# 评估
lr_evaluator = RegressionEvaluator(metricName="r2", predictionCol='prediction', labelCol='medv')
r2 = lr_evaluator.evaluate(predictions)
test_evaluation = lrModel.evaluate(test_data)
print("r2: %f" % r2)
print("RMSE: %f" % test_evaluation.rootMeanSquaredError)
