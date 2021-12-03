# FlightDelaySpark
Spark application that creates a machine learning model for predicting the arrival delay of commercial flights

###How to execute the script
First, install the requirements 

<code>python -m pip install -r requirements.txt</code>

For local, execute <code>spark-submit -master local[*] FlightDelay.py PATH</code>
Where PATH is the location of CVS files. 
You can find them here: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7

###Execution example
<code>
%SPARK_HOME%\bin\spark-submit --master local[*] FlightDelay.py file:///C:\UPM\big_data_assignments\data\2000
</code>