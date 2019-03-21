from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator

# 定义spark
spark = SparkSession \
    .builder \
    .appName("LRExample") \
    .getOrCreate()


# 加载数据
data = spark.read.format("libsvm")\
    .load("file:///D:\Spark\spark-2.3.3-bin-hadoop2.7\data\mllib\sample_libsvm_data.txt")
data.show(10)


# 划分数据集测试集
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]


# 训练模型
lr = LinearRegression(maxIter=10000, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(train)


print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# 预测
predictions = lrModel.transform(test)
predictions.show()

# 评估
lr_evaluator = RegressionEvaluator(metricName="r2", predictionCol='prediction', labelCol='label')
r2 = lr_evaluator.evaluate(predictions)
test_evaluation = lrModel.evaluate(test)
print("r2: %f" % r2)
print("RMSE: %f" % test_evaluation.rootMeanSquaredError)
