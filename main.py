from trainer import SparkConfig, Trainer
from model import SVM, RandomForest, LogisticRegressionModel
import socket
import json
import numpy as np
import sys


def test(TCP_IP, TCP_PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    print(f"[Receiver] Connected to {TCP_IP}:{TCP_PORT}")

    buffer = ""
    try:
        while True:
            data = s.recv(4096).decode()
            if not data:
                break

            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1) 
                try:
                    payload = json.loads(line)
                    print(f"[Receiver] Received batch of {len(payload)} samples")

                    labels = [v["label"] for v in payload.values()]
                    print(f"[Receiver] Labels: {labels[:10]}...")

                except json.JSONDecodeError as e:
                    print(f"[Receiver] Failed to decode JSON: {e}")

    except KeyboardInterrupt:
        print("\n[Receiver] Interrupted by user.")
    finally:
        s.close()
        print("[Receiver] Socket closed.")




if __name__ == "__main__":
    spark_config = SparkConfig()

    if len(sys.argv) < 2:
        print("Usage: python main.py [model_name]")
        print("Available models: svm, random_forest, logistic_regression")
        sys.exit(1)

    model_name = sys.argv[1]

    if model_name == "svm":
        model = SVM(loss="hinge", penalty="l2", max_iter=100, tol=1e-4)
    elif model_name == "random_forest":
        model = RandomForest(num_trees=100, max_depth=10, seed=42)
    elif model_name == "logistic_regression":
        model = LogisticRegressionModel(max_iter=100, reg_param=0.3, elastic_net_param=0.8)
    else:
        print(f"Invalid model name: {model_name}")
        print("Available models: svm, random_forest, logistic_regression")
        sys.exit(1)

    trainer = Trainer(model, "train", spark_config)
    trainer.train()
    # trainer.predict()
