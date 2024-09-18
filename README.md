
# COVID-19 Chest X-Ray Image Classification

This project demonstrates a COVID-19 chest X-ray image classification model using a Convolutional Neural Network (CNN) architecture. The model is designed to classify chest X-ray images into two categories: **Normal** and **COVID-19**.

## Dataset
The dataset consists of chest X-ray images split into two categories:
- Normal
- COVID-19

The dataset is divided into training and testing sets, and further augmented to improve model generalization.

## Model Architecture
The Convolutional Neural Network (CNN) model is built using TensorFlow and Keras. The architecture includes the following layers:
1. **Convolutional Layers**: Three convolutional layers with 32 filters and ReLU activation.
2. **Batch Normalization**: Applied after each convolutional layer.
3. **MaxPooling**: Applied after each batch normalization to reduce spatial dimensions.
4. **Dense Layers**: Two dense layers with ReLU activation for classification.
5. **Dropout Layers**: Used to prevent overfitting.
6. **Output Layer**: A single dense layer with a sigmoid activation function for binary classification (Normal or COVID-19).

## Data Augmentation
Data augmentation techniques are applied to the training data to improve the robustness of the model. Augmentation includes:
- Rotation
- Brightness adjustment
- Image flipping
- Image blurring

## Model Training
The model is compiled using the **RMSprop** optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric. Training is performed for 50 epochs.

The class weights are adjusted to account for any imbalance in the dataset.

## Results
The model's performance is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix on the test set. A detailed classification report is generated to assess the model's performance.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yazidramdhani/covid19-classification.git
    cd covid19-classification
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the dataset from Kaggle:
    ```bash
    kaggle datasets download -d kamildinleyici/cov-dataset
    ```
4. Unzip the dataset and place it in the appropriate directories as mentioned in the notebook.

## Usage
1. Run the training script in the Jupyter notebook or Python script to train the model.
2. Evaluate the model on the test set.
3. Use the model to predict COVID-19 cases from new chest X-ray images.

## Example Inference
Upload a chest X-ray image and use the trained model to predict if the image belongs to a COVID-19 positive patient.

```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
test_img = image.load_img('path_to_image', target_size=(150, 150), color_mode='grayscale')
pp_test_img = image.img_to_array(test_img) / 255.0
pp_test_img = np.expand_dims(pp_test_img, axis=0)

# Make a prediction
prediction = model.predict(pp_test_img)

# Output the result
if prediction >= 0.5:
    print("COVID-19 Detected")
else:
    print("Normal Case Detected")
```
