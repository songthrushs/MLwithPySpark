#!/usr/bin/env python
# coding: utf-8

# installing libraries
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.functions import col

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline 
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor

from pyspark.ml.evaluation import RegressionEvaluator

# initiating spark

spark = SparkSession.builder.master("local").appName('A2_Spark').getOrCreate()
sc = spark.sparkContext

sqlContext = SQLContext(spark.sparkContext)

# reading data, and splitting as features and labels

df = spark.read.csv('white_wine_corr.csv', inferSchema=True, header=True)
df = df.drop('_c0')

X = df.drop('quality')
y = df.select(col('quality'))

# Up-sampling the data

X_Pandas = X.toPandas()
y_Pandas = y.toPandas()

X_train, X_test, y_train, y_test = train_test_split(X_Pandas, y_Pandas, test_size=0.2, random_state=0)

ss = {3: 3000, 4: 3000, 5: 3000, 6: 3000, 7: 3000, 8: 3000, 9: 3000}
sm = SMOTE(sampling_strategy=ss, random_state=12, k_neighbors=4)
X_train, y_train = sm.fit_resample(X_train, y_train)

df_1 = pd.DataFrame(X_train, columns=['fixed_acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                                      'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                      'alcohol'])
df_2 = pd.DataFrame(y_train, columns=['quality'])

#rdd = sc.parallelize([df_1, df_2], numSlices=100000)

df_new = df_1.combine_first(df_2)
df_new = spark.createDataFrame(df_new)

print('Total number of rows increased to :  ', df_new.count())

# Vector Assembler

assembler = VectorAssembler(inputCols=['fixed_acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH',
                                       'sulphates', 'alcohol'], outputCol='features')

df_output = assembler.transform(df_new)

# Normalizing the data

standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
scaled_df = standardScaler.fit(df_output).transform(df_output)

# Linear Regression

train_data, test_data = scaled_df.randomSplit([.8, .2])

lr = (LinearRegression(featuresCol='features', labelCol='quality', predictionCol='pred_quality', standardization=True))
linearModel = lr.fit(scaled_df)

predictions = linearModel.transform(test_data)
predictions.select("pred_quality", "quality").show()

evaluator = RegressionEvaluator(labelCol="quality", predictionCol="pred_quality", metricName="r2")
r_square = evaluator.evaluate(predictions)
print(" R^2 on test data for LINEAR REGRESSION = %g" % r_square)

# Hyper-parameter Tuning for Linear Regression

pipeline = Pipeline(stages=[lr])

paramGrid = ParamGridBuilder().addGrid(lr.aggregationDepth, [2, 3]).addGrid(lr.elasticNetParam, [0.0, 0.05, 0.1])     \
    .addGrid(lr.maxIter, [10, 100]).addGrid(lr.epsilon, [1.2, 1.4]).addGrid(lr.regParam, [0.0, 0.1, 0.03]).build()

cross_val = CrossValidator(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator().setLabelCol("quality").setPredictionCol("pred_quality"),
                           numFolds=6)

cvModel = cross_val.fit(train_data)
print("Best model parameters", cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])

predictions = cvModel.transform(test_data)
predictions.select("pred_quality", "quality").show()

r_square = evaluator.evaluate(predictions)
print(" R^2 on test data for LINEAR REGRESSION_with_hyper-parameter = %g" % r_square)

# Random Forest Regressor

rf = RandomForestRegressor(featuresCol='features', labelCol='quality', predictionCol='pred_quality')
pipeline = Pipeline(stages=[rf])

rf_model = pipeline.fit(train_data)

predictions = rf_model.transform(test_data)
predictions.select("pred_quality", "quality").show()

r_square = evaluator.evaluate(predictions)
print(" R^2 on test data for RANDOM FOREST REGRESSION= %g" % r_square)

# Hyper-parameter Tuning for Random Forest Regressor

pipeline = Pipeline(stages=[rf])

paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 30]).build()

cross_val = CrossValidator(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator().setLabelCol("quality").setPredictionCol("pred_quality"),
                           numFolds=2)

cvModel = cross_val.fit(train_data)
print(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])

predictions = cvModel.transform(test_data)
predictions.select("pred_quality", "quality").show()

r_square = evaluator.evaluate(predictions)
print(" R^2 on test data for RANDOM FOREST REGRESSION = %g" % r_square)
