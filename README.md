This respository demonstrates how to stimulate the process of sending and receiving streamning data within a device.
The dataset: Cifar10
There are three Machine learning models that you can choose: SVM, Random forest, and Logistic regression. Remember choose one.
To create a sender: spark-submit stream.py --folder cifar-10-batches-py --batch-size [32]
To create a listener and train an opitional model: python main.py [svm]
