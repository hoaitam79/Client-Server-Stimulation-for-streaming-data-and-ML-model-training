import pyspark

from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT

class SparkConfig:
    appName = "CIFAR"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 6100
    batch_interval = 2

from dataloader import DataLoader
class Trainer:
    def __init__(self, 
                 model, 
                 split:str, 
                 spark_config:SparkConfig
                 ) -> None:

        self.model = model
        self.split = split
        self.sparkConf = spark_config
        #self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]",f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc, self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc, self.ssc, self.sqlContext, self.sparkConf)
        self.sc.setLogLevel("ERROR")

    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        try:
            print(f"[Spark] New RDD at {timestamp}")
            if not rdd.isEmpty():
                count = rdd.count()
                print(f"[Spark] RDD has {count} records")
                df = rdd.map(lambda x: (x[:-1], int(x[-1]))).toDF(["image", "label"])
                preds, acc, prec, rec, f1 = self.model.train(df)
                print(
                    f"[Spark] Train metrics - Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            else:
                print("[Spark] Empty RDD")
        except Exception as e:
            print(f"[Spark ERROR] Exception in __train__: {e}")


    def predict(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__predict__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __predict__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        try:
            print(f"[Spark] New RDD at {timestamp}")
            if not rdd.isEmpty():
                count = rdd.count()
                print(f"[Spark] RDD has {count} records")

                sample = rdd.take(1)
                print(f"[Spark] Sample: {sample}")
                print(f"[Spark] Sample type: {type(sample[0])} | len = {len(sample[0])}")

                df = rdd.map(lambda x: (x[:-1], int(x[-1]))) \
                    .toDF(["image", "label"])

                preds, acc, prec, rec, f1 = self.model.predict(df)

                print(
                    f"[Spark] Predict metrics - Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            else:
                print("[Spark] Empty RDD")

        except Exception as e:
            print(f"[Spark ERROR] Exception in __predict__: {e}")
    #     if not rdd.isEmpty():
    #         schema = StructType([
    #             StructField(name="image", dataType=VectorUDT(), nullable=True),
    #             StructField(name="label",dataType=IntegerType(),nullable=True)])
            
    #         df = self.sqlContext.createDataFrame(rdd, schema)
            
    #         accuracy, loss, precision, recall, f1, cm = self.model.predict(df, self.raw_model)
    #         self.cm += cm
    #         self.test_accuracy += accuracy/total_batches
    #         self.test_loss += loss/total_batches
    #         self.test_precision += precision/total_batches
    #         self.test_recall += recall/total_batches
    #         self.test_f1 += f1/total_batches
    #         print(f"Test Accuracy : ", self.test_accuracy)
    #         print(f"Test Loss : ",self.test_loss)
    #         print(f"Test Precision :",self.test_precision)
    #         print(f"Test Recall : ", self.test_recall)
    #         print(f"Test F1 Score: ",self.test_f1)
    #         print(f"Confusion matrix: \n", self.cm)

    #     print(f"batch: {self.batch_count}")
    #     print("Total Batch Size of RDD Received :",rdd.count())
    #     print("---------------------------------------")   
        
        