from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2
import matplotlib.pyplot as plt
############################################################################

model = load_model('Path_of_DenseNet_model')
print(model.summary())
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    directory= 'Test_Set_Path',
    target_size=(64,64),
    color_mode="grayscale",
    batch_size=1,
    shuffle = False)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator , steps=STEP_SIZE_TEST , verbose=1)
predicted_class_indices=np.argmax(pred, axis=1)



print("Predicted")
print(predicted_class_indices)
print("Test Generator Classses")
print(test_generator.classes)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, predicted_class_indices))
print(accuracy_score(test_generator.classes, predicted_class_indices))










