# FlightDelaySpark
Spark application that creates a machine learning model for predicting the arrival delay of commercial flights

### How to execute the script
First, install the requirements 

<code>python -m pip install -r requirements.txt</code>

For local, execute <code>spark-submit -master local[*] FlightDelay.py PATH SAMPLE LOG</code>

- PATH is the location of CVS files; default is data/*.csv
- SAMPLE is the fraction [0-1] for sampling the original data set, 0.1 is 10%; default: 1.0 (100%)
- LOG is the Log level: INFO, WARN, ERROR; default: WARN

You can find data here: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7

### Execution example
- Execution example with PATH

<code>
%SPARK_HOME%\bin\spark-submit --master local[*] FlightDelay.py file:///C:\UPM\big_data_assignments\data\2000
</code>

- Execution example with PATH and SAMPLE 

<code>
%SPARK_HOME%\bin\spark-submit --master local[*] FlightDelay.py file:///C:\UPM\FlightDelaySpark\data 0.1
</code>

- Execution example with PATH, SAMPLE and LOG

<code>
%SPARK_HOME%\bin\spark-submit --master local[*] FlightDelay.py file:///C:\UPM\FlightDelaySpark\data 0.05 ERROR
</code>