import sys
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import Bucketizer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline


def main(spark, path="data/*.csv", sample=1.0, log='WARN'):

    print('**** Parameters ****')
    print('Path: ', path)
    print('Sample: ', sample)
    print('Log level: ', log)

    print('Searching files in: ', path)
    spark.sparkContext.setLogLevel(log)

    # Forbidden variables
    forbidden = ('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay',
                 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')

    # we consider these variables as irrelevant
    # They don't give us relevant information
    irrelevants = ('Year', 'DayofMonth', 'FlightNum', 'TailNum', 'CancellationCode', 'Dest')

    # we consider these variables as redundant information in the data set
    # They are represented in other variables; DepTime = CRSDepTime - DepDelay
    redundants = ('DepTime', 'CRSElapsedTime')

    flightsDF = spark.read.csv(path, header=True).drop(*forbidden + irrelevants + redundants)

    # dummy = ('Month', 'DayOfWeek', 'UniqueCarrier')
    # toDiscretization = ('CRSDepTime', 'CRSArrTime', 'Distance', 'Origin')
    # toInteger = ('ArrDelay', 'DepDelay', 'TaxiOut', 'Cancelled',)

    # We cast all numerical and boolean variables
    flightsDF = flightsDF.withColumn('CRSDepTime', col('CRSDepTime').cast('integer'))
    flightsDF = flightsDF.withColumn('CRSArrTime', col('CRSArrTime').cast('integer'))
    flightsDF = flightsDF.withColumn('Distance', col('Distance').cast('integer'))
    flightsDF = flightsDF.withColumn('ArrDelay', col('ArrDelay').cast('integer'))
    flightsDF = flightsDF.withColumn('DepDelay', col('DepDelay').cast('integer'))
    flightsDF = flightsDF.withColumn('TaxiOut', col('TaxiOut').cast('integer'))
    flightsDF = flightsDF.withColumn('Cancelled', col('Cancelled').cast('boolean'))

    # We have missing values, most of them (99%) it's because of cancelled flights
    # we remove all the cancelled flights, the remaining missing values are deleted too
    flightsDF = flightsDF.filter(col("Cancelled") == False)
    flightsDF = flightsDF.drop('Cancelled')
    flightsDF = flightsDF.na.drop()

    if sample > 0.0 and sample < 1.0:
        flightsDF = flightsDF.sample(withReplacement=True, fraction=sample)

    # Discretization of Departures and Arrivals => Transforming Departures times in morning, afternoon, evening and night
    splitHours = [-float('inf'), 600, 1200, 1800, float('inf')]
    discretizationCRSDepTime = Bucketizer(splits=splitHours, inputCol='CRSDepTime', outputCol='CRSDepTimeCat')
    discretizationCRSArrTime = Bucketizer(splits=splitHours, inputCol='CRSArrTime', outputCol='CRSArrTimeCat')

    # Discretization of Distance => Transforming Distances in short, medium and large
    splitDistances = [-float('inf'), 500, 1500, float('inf')]
    discretizationDistance = Bucketizer(splits=splitDistances, inputCol='Distance', outputCol='DistanceCat')

    # This pipeline performs the discretization steps
    pipelineDiscretization = Pipeline(stages=[discretizationCRSDepTime, discretizationCRSArrTime, discretizationDistance])
    pipelineModelDiscretization = pipelineDiscretization.fit(flightsDF)
    flightsDF = pipelineModelDiscretization.transform(flightsDF)

    # Discretization of Origin (airport) => According to number of flights small, medium, large, big
    # flightsDF = flightsDF.withColumn("AirportSize", F.count(col('Origin')).over(Window.partitionBy(flightsDF.Origin)))
    # splitSize = [-float('inf'), 25000, 50000, 150000, float('inf')]
    # bucketizer = Bucketizer(splits=splitSize, inputCol='OriginSize', outputCol='OriginSizeCat')
    # flightsDF = bucketizer.transform(flightsDF)

    # Discretization of Origins (airport) => According to number of flights small, medium, large and big
    # First, we compute the total amount of flights in every Origin, then we classify them
    airportsDF = flightsDF.groupBy('Origin').agg(F.count(col('Origin')).alias('OriginSize'))
    splitSize = [-float('inf'), 25000, 50000, 150000, float('inf')]
    bucketizer = Bucketizer(splits=splitSize, inputCol='OriginSize', outputCol='OriginSizeCat')
    airportsDF = bucketizer.transform(airportsDF)
    flightsDF = flightsDF.join(airportsDF, 'Origin')
    flightsDF = flightsDF.drop('CRSDepTime', 'CRSArrTime', 'Distance', 'Origin', 'OriginSize')

    # we can delete the airportDF
    airportsDF.unpersist()

    # Split data in training and testing
    split = flightsDF.randomSplit([0.7, 0.3], seed=132)
    training = split[0]
    tests = split[1].randomSplit([0.5, 0.5], seed=132)
    comparingModels = tests[0]
    test = tests[1]

    # Transforming categorical variables to numeric
    indexer = StringIndexer(
        inputCols=['Month', 'DayOfWeek', 'UniqueCarrier', 'CRSDepTimeCat', 'CRSArrTimeCat', 'DistanceCat', 'OriginSizeCat'],
        outputCols=['MonthNum', 'DayOfWeekNum', 'UniqueCarrierNum', 'CRSDepTimeNum', 'CRSArrTimeNum', 'DistanceNum', 'OriginSizeNum'])

    # ONE-HOT encoding for categorical variables
    encoder = OneHotEncoder(inputCols=['MonthNum', 'DayOfWeekNum', 'UniqueCarrierNum', 'CRSDepTimeNum', 'CRSArrTimeNum', 'DistanceNum', 'OriginSizeNum'],
                            outputCols=['MonthOH', 'DayOfWeekOH', 'UniqueCarrierOH', 'CRSDepTimeOH', 'CRSArrTimeOH', 'DistanceOH', 'OriginSizeOH'],
                            dropLast=True)

    # Selection of variables for the models
    features = ['DepDelay', 'TaxiOut', 'DistanceOH', 'CRSDepTimeOH', 'OriginSizeOH', 'CRSArrTimeOH', 'MonthOH',
                'DayOfWeekOH', 'UniqueCarrierOH']
    assembler = VectorAssembler(inputCols=features, outputCol='features')

    # Defining Regression Models
    lr = LinearRegression(featuresCol='features', labelCol='ArrDelay', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    glr = GeneralizedLinearRegression(family="gaussian", link="identity", labelCol='ArrDelay', maxIter=10, regParam=0.3)

    # Defining the pipelines to training data for models
    LRPipeline = Pipeline(stages=[indexer, encoder, assembler, lr])
    GLRPipeline = Pipeline(stages=[indexer, encoder, assembler, glr])

    # Grids of hyperparameters used to find the best
    paramGridLR = ParamGridBuilder() \
        .addGrid(lr.maxIter, [25, 100]) \
        .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
        .build()

    paramGridGLR = ParamGridBuilder() \
        .addGrid(glr.maxIter, [25, 100]) \
        .addGrid(glr.regParam, [0.1, 0.01, 0.001]) \
        .build()

    # Defining the train-validation
    trainValidationLR = TrainValidationSplit(estimator=LRPipeline,
                               estimatorParamMaps=paramGridLR,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay'),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    trainValidationGLR = TrainValidationSplit(estimator=GLRPipeline,
                               estimatorParamMaps=paramGridGLR,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay'),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    # Finding the best model for Linear Regression using training data
    print('**** Computing Linear Regression Model ****')
    tvModelLR = trainValidationLR.fit(training)

    print('**** Best Linear Regression Model on Training ****')
    bestModelLR = tvModelLR.bestModel.stages[-1]
    trainingSummaryLR = bestModelLR.summary
    print("Coefficients: ", str(bestModelLR.coefficients))
    print("Intercept: ", str(bestModelLR.intercept))
    print("regParam: ", bestModelLR._java_obj.getRegParam())
    print("maxIter: ", bestModelLR._java_obj.getMaxIter())
    print("elasticNetParam", bestModelLR._java_obj.getElasticNetParam())
    print("RMSE: ", trainingSummaryLR.rootMeanSquaredError)
    print("MSE: ", trainingSummaryLR.meanSquaredError)
    print("MAE: ", trainingSummaryLR.meanAbsoluteError)
    print("R2: ", trainingSummaryLR.r2)

    # Finding the best model for Generalized Regression Model using training data
    print('**** Computing Generalized Regression Model ****')
    tvModelGLR = trainValidationGLR.fit(training)

    print('**** Best Generalized Regression Model on Training ****')
    bestModelGLR = tvModelGLR.bestModel.stages[-1]
    print("Coefficients: " + str(bestModelGLR.coefficients))
    print("Intercept: " + str(bestModelGLR.intercept))
    print("regParam: ", bestModelGLR._java_obj.getRegParam())
    print("maxIter: ", bestModelGLR._java_obj.getMaxIter())
    trainingSummaryGLR = bestModelGLR.summary
    print(trainingSummaryGLR)

    # Compare models with new observations data comparingModels
    # We check what model has less RMSE for new observation
    print('**** Comparing the Models ****')
    evaluatorRMSE = RegressionEvaluator(metricName="rmse", labelCol='ArrDelay', predictionCol='prediction')
    evaluatorR2 = RegressionEvaluator(metricName="r2", labelCol='ArrDelay', predictionCol='prediction')

    predictionLR = tvModelLR.transform(comparingModels)
    predictionGLR = tvModelGLR.transform(comparingModels)

    # evaluate the model R2 and RMSE
    R2LR = evaluatorR2.evaluate(predictionLR)
    RMSELR = evaluatorRMSE.evaluate(predictionLR)
    R2GLR = evaluatorR2.evaluate(predictionGLR)
    RMSEGLR = evaluatorRMSE.evaluate(predictionGLR)
    print('**** Results for the Linear Regression Model ****')
    print('RMSE: ', RMSELR)
    print('R2: ', R2LR)

    print('**** Results for the Generalized Regression Model ****')
    print('RMSE: ', RMSEGLR)
    print('R2: ', R2GLR)


    # Selecting the best model by comparin the RMSE,
    # The best model is which minimizes the RMSE
    if RMSELR <= RMSEGLR:
        print('The best model is the Linear Regression Model')
        bestModel = tvModelLR
    else:
        print('The best model is the Generalized Regression Model')
        bestModel = tvModelGLR

    # Compute the predictions for the test data
    print('**** Results for Testing Data Using the best model ****')
    predictions = bestModel.transform(test)

    RMSE = evaluatorRMSE.evaluate(predictions)
    R2 = evaluatorR2.evaluate(predictions)
    print('RMSE: ', RMSE)
    print('R2: ', R2)

    spark.stop()


if __name__ == "__main__":
    # Arguments
    # path: The path for CSV files; default: data/*.csv
    # sample: The fraction [0-1] for sampling the data set; default: 1.0
    # log level: Log level: INFO, WARN, ERROR; default: WARN
    print(sys.argv)
    p = sys.argv[1] if len(sys.argv) >= 2 else "data/*.csv"
    s = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0
    l = sys.argv[3] if len(sys.argv) >= 4 else 'WARN'
    print(p,s,l)
    main(SparkSession.builder.appName("FlightDelay").getOrCreate(), p, s, l)
