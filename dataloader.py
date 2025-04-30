import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector
import torchvision.transforms as T
from transforms_scr import ApplyTransform
from trainer import SparkConfig
from PIL import Image
import json

def transform_image(x):
    if x is None or len(x) != 3073:
        print("error data:", x)
        return None

class DataLoader:
    def __init__(self, 
                 sparkContext:SparkContext, 
                 sparkStreamingContext: StreamingContext, 
                 sqlContext: SQLContext,
                 sparkConf: SparkConfig) -> None:
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host, 
            port=self.sparkConf.port
        )

    def parse_stream(self) -> DStream:
        raw_stream = self.stream
        parsed_stream = raw_stream.map(lambda line: json.loads(line))
        parsed_stream = parsed_stream.flatMap(lambda x: x.values())
        parsed_stream = parsed_stream.map(
                lambda x: [x[f"feature-{i}"] for i in range(3072)] + [x["label"]]
        )
        pixels = parsed_stream.map(
            lambda x: [np.array(x[:-1], dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0), x[-1]]
        )
        pixels = parsed_stream
        return pixels
