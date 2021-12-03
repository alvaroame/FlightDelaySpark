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


def main(spark):
    path = sys.argv[1]
    print('Searching files in ', path)

    #Forbidden variables
    forbidden = ('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay',
                 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')

    #we consider these variables as irrelevant
    irrelevants = ('Year', 'DayofMonth', 'FlightNum', 'TailNum', 'CancellationCode','Dest')

    #we consider these variables as redundant information in the data set
    redundants = ('DepTime','CRSElapsedTime')

    flightsDF = spark.read.csv(path, header=True)\
        .drop(*forbidden+irrelevants+redundants)

    #dummy = ('Month', 'DayOfWeek', 'UniqueCarrier')
    #toDiscretazion = ('CRSDepTime', 'CRSArrTime', 'Distance', 'Origin')
    #toInteger = ('ArrDelay', 'DepDelay', 'TaxiOut', 'Cancelled',)

    flightsDF = flightsDF.withColumn('CRSDepTime', col('CRSDepTime').cast('integer'))
    flightsDF = flightsDF.withColumn('CRSArrTime', col('CRSArrTime').cast('integer'))
    flightsDF = flightsDF.withColumn('Distance', col('Distance').cast('integer'))
    flightsDF = flightsDF.withColumn('ArrDelay', col('ArrDelay').cast('integer'))
    flightsDF = flightsDF.withColumn('DepDelay', col('DepDelay').cast('integer'))
    flightsDF = flightsDF.withColumn('TaxiOut', col('TaxiOut').cast('integer'))
    flightsDF = flightsDF.withColumn('Cancelled', col('Cancelled').cast('integer'))

    flightsDF = flightsDF.withColumn('Month', col('Month').cast('integer'))
    flightsDF = flightsDF.withColumn('DayOfWeek', col('DayOfWeek').cast('integer'))

    flightsDF = flightsDF.filter(col("Cancelled") == 0)
    flightsDF = flightsDF.drop('Cancelled')
    flightsDF = flightsDF.na.drop()

    #Discretation of CRSDepTime => Transforming Departures times in morning, afternoon, evening and night
    splits_hours = [-float('inf'), 600, 1200, 1800, float('inf')]
    bucketizer = Bucketizer(splits=splits_hours, inputCol='CRSDepTime', outputCol='CRSDepTimeCat')
    flightsDF = bucketizer.transform(flightsDF)
    bucketizer = Bucketizer(splits=splits_hours, inputCol='CRSArrTime', outputCol='CRSArrTimeCat')
    flightsDF = bucketizer.transform(flightsDF)

    #Discretation of Distance => Transforming Distances in short, medium, large
    splits_distances = [-float('inf'), 500, 1500, float('inf')]
    bucketizer = Bucketizer(splits=splits_distances, inputCol='Distance', outputCol='DistanceCat')
    flightsDF = bucketizer.transform(flightsDF)

    #Discretation of Origin => According to number of flight in the airport small-hub, medium, larga, big
    flightsDF = flightsDF.withColumn("SizeOfOrigin",F.count(col('Origin')).over(Window.partitionBy(flightsDF.Origin)))
    splits_size = [-float('inf'), 25000, 50000, 150000, float('inf')]
    bucketizer = Bucketizer(splits=splits_size, inputCol='SizeOfOrigin', outputCol='SizeOfOriginCat')
    flightsDF = bucketizer.transform(flightsDF)

    #airportsDF = flightsDF.groupBy('Origin').agg(F.count(col('Origin')).alias('SizeOfOrigin'))
    #splits_size = [-float('inf'), 25000, 50000, 150000, float('inf')]
    #bucketizer = Bucketizer(splits=splits_size, inputCol='SizeOfOrigin', outputCol='SizeOfOriginCat')
    #airportsDF = bucketizer.transform(airportsDF)

    #flightsDF = flightsDF.join(airportsDF, 'Origin')

    # Transforming categorical variables to numeric
    indexer = StringIndexer(inputCol="UniqueCarrier", outputCol="UniqueCarrierNum")
    flightsDF = indexer.fit(flightsDF).transform(flightsDF)

    # ONE-HOT encoding for categorical variables
    encoder = OneHotEncoder(inputCols=['Month', 'DayOfWeek', 'UniqueCarrierNum', 'CRSDepTimeCat', 'CRSArrTimeCat', 'DistanceCat', 'SizeOfOriginCat'],
                            outputCols=['MonthOH', 'DayOfWeekOH', 'UniqueCarrierOH', 'CRSDepTimeOH', 'CRSArrTimeOH', 'DistanceOH', 'SizeOfOriginOH'])
    flightsDF = encoder.fit(flightsDF).transform(flightsDF)

    #Selection of variables
    features = ['DepDelay', 'TaxiOut', 'MonthOH', 'DayOfWeekOH', 'UniqueCarrierOH', 'CRSDepTimeOH', 'CRSArrTimeOH','DistanceOH', 'SizeOfOriginOH']
    vectorAssembler = VectorAssembler(inputCols=features, outputCol='features')

    vdf_sel = vectorAssembler.transform(flightsDF)
    vdf_sel = vdf_sel.select(['features', 'ArrDelay'])

    # split dataframes
    splits = vdf_sel.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]

    lr = LinearRegression(featuresCol='features', labelCol='ArrDelay', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_df)
    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    spark.stop()


if __name__ == "__main__":
    main(SparkSession.builder.appName("FlightDelay").getOrCreate())
