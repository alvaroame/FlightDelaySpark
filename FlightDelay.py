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
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline

def main(spark):
    spark.sparkContext.setLogLevel('INFO')

    path = sys.argv[1]
    print('Searching files in: ', path)

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
    # bucketizer = Bucketizer(splits=splitSize, inputCol='AirportSize', outputCol='AirportSizeCat')
    # flightsDF = bucketizer.transform(flightsDF)

    # Discretization of Origins (airport) => According to number of flights small, medium, large and big
    # First, we compute the total amount of flights in every Origin, then we classify them
    airportsDF = flightsDF.groupBy('Origin').agg(F.count(col('Origin')).alias('AirportSize'))
    splitSize = [-float('inf'), 25000, 50000, 150000, float('inf')]
    bucketizer = Bucketizer(splits=splitSize, inputCol='AirportSize', outputCol='AirportSizeCat')
    airportsDF = bucketizer.transform(airportsDF)
    flightsDF = flightsDF.join(airportsDF, 'Origin')
    flightsDF = flightsDF.drop('CRSDepTime', 'CRSArrTime', 'Distance', 'Origin', 'AirportSize')

    # we can delete the airport DF
    airportsDF.unpersist()

    # Split data in training and testing
    split = flightsDF.randomSplit([0.7, 0.3], seed=132)
    training = split[0]
    test = split[1]

    # Transforming categorical variables to numeric
    indexer = StringIndexer(
        inputCols=['Month', 'DayOfWeek', 'UniqueCarrier', 'CRSDepTimeCat', 'CRSArrTimeCat', 'DistanceCat', 'AirportSizeCat'],
        outputCols=['MonthNum', 'DayOfWeekNum', 'UniqueCarrierNum', 'CRSDepTimeNum', 'CRSArrTimeNum', 'DistanceNum', 'AirportSizeNum'])

    # ONE-HOT encoding for categorical variables
    encoder = OneHotEncoder(inputCols=['MonthNum', 'DayOfWeekNum', 'UniqueCarrierNum', 'CRSDepTimeNum', 'CRSArrTimeNum', 'DistanceNum', 'AirportSizeNum'],
                            outputCols=['MonthOH', 'DayOfWeekOH', 'UniqueCarrierOH', 'CRSDepTimeOH', 'CRSArrTimeOH', 'DistanceOH', 'AirportSizeOH'],
                            dropLast=False)

    # Selection of variables
    features = ['DepDelay', 'TaxiOut', 'DistanceOH', 'CRSDepTimeOH', 'AirportSizeOH', 'CRSArrTimeOH', 'MonthOH',
                'DayOfWeekOH', 'UniqueCarrierOH']
    assembler = VectorAssembler(inputCols=features, outputCol='features')

    print('Linear Regression Model')
    # Implementing a Linear Regression Model
    lr = LinearRegression(featuresCol='features', labelCol='ArrDelay', maxIter=10,
                          regParam=0.3, elasticNetParam=0.8)

    # Fit the pipeline to training data
    regressionPipeline = Pipeline(stages=[indexer, encoder, assembler, lr])

    paramGrid = ParamGridBuilder() \
        .addGrid(lr.maxIter, [10, 50, 100]) \
        .addGrid(lr.regParam, [0.3, 0.1, 0.01]) \
        .build()

    crossval = CrossValidator(estimator=regressionPipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=RegressionEvaluator(labelCol='ArrDelay'),
                              numFolds=4)

    cvModel = crossval.fit(training)

    # Make predictions on test data
    prediction = cvModel.transform(test)

    print('****Linear Regression Model ****')
    bestModel = cvModel.bestModel.stages[-1]
    trainingSummary = bestModel.summary
    print("Coefficients: ", str(bestModel.coefficients))
    print("Intercept: ", str(bestModel.intercept))
    print("RMSE: ", trainingSummary.rootMeanSquaredError)
    print("R2: ", trainingSummary.r2)
    print("regParam: ", bestModel._java_obj.getRegParam())
    print("maxIter: ", bestModel._java_obj.getMaxIter())
    print("elasticNetParam", bestModel._java_obj.getElasticNetParam())

    # evaluate the model
    evaluatorR2 = RegressionEvaluator(metricName="r2", labelCol='ArrDelay', predictionCol='prediction')
    evaluatorRMSE = RegressionEvaluator(metricName="rmse", labelCol='ArrDelay', predictionCol='prediction')
    R2 = evaluatorR2.evaluate(prediction)
    RMSE = evaluatorRMSE.evaluate(prediction)
    print('****Evaluation of the model on test ****')
    print('R2: ', R2)
    print('RMSE: ', RMSE)

    spark.stop()


if __name__ == "__main__":
    main(SparkSession.builder.appName("FlightDelay").getOrCreate())
