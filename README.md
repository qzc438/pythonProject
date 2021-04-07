# pythonProject
Provide deep learning model set and compiler for JaveEE project
## File Description
### config.py: Switch different deep learning models and backends  
Current deep learning models available: CNN, LSTM  
Current backends available: Keras, Pyotrch, TensorFlow
### data_file.py:
Data file location
### data_load.py:
Data load function
### data_review.py:
Explore the dataset 
### flask_webservice.py: 
Web service, flask server, input: dataset & model name, output: performance
### main.py:
Main function
### main_kfold.py: 
K fold cross-validation with the dataset
### main_onnx.py:
Translate saved model file to .onnx file
### main_original_keras.py:
Original file using Keras
### main_original_pytorch.py:
Original file using Pytorch
### main_original_tensorflow.py:
Original file using TensorFlow
### main-split.py:
Manually separete train and test dataset
### model.py:
Model architecture
### pytorch_minst.py:
Minst dataset using Pytorch
### util.py:
Utilities
