# AudioClassification
Custom Audio Classification using CNN

#Introduction:
In this project, we explore the use of Convolutional Neural Networks (CNNs) to classify audio files based on the emotions they convey. Emotion classification can be useful in a variety of applications, including speech recognition, automatic music tagging, and sentiment analysis. The project aims to classify audio data into different categories using a Convolutional Neural Network (CNN). The input audio files are processed using the Librosa library, and MFCC features are extracted from them. These features are then used to train a CNN model to classify the audio files into different categories.

#Data
 The dataset used in this project is the Toronto emotional speech set (TESS) dataset, which contains 2800 audio files from 2 actors, each uttering 200 target words in 7 different emotions: angry, disgust, fear, happy, neutral, pleasant surprised and sad. The data is loaded using the Librosa library, and MFCC features are extracted from each audio file. The labels for each audio file are extracted from the folder names.

Dataset (TESSS) - (https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

#Preprocessing:
The extracted MFCC features are normalized and reshaped for input to the CNN model. The data is then split into training and testing sets, with 80% of the data used for training and 20% for testing.

#Model Architecture:
The CNN model consists of two Conv1D layers with ReLU activation functions and MaxPooling1D layers to downsample the output. The output of the second MaxPooling1D layer is flattened and passed through a Dense layer with a ReLU activation function. A dropout layer is added to reduce overfitting, and the output is passed through a final Dense layer with a Softmax activation function to classify the input data into different categories.

#Model Training:
The model is trained on the training set using categorical cross-entropy as the loss function, Adam as the optimizer, and accuracy as the evaluation metric. The labels for each audio file are one-hot encoded before training the model.

#Results:
The trained model achieved an accuracy of 97% on the test set. The model can be used to predict the category of any new audio file after extracting MFCC features and reshaping them in the same way as the training data.

#Conclusion:
In conclusion, this project shows how to use MFCC features and a CNN model to classify audio data into different categories. The model achieved good accuracy on the test set, indicating that it can be used for practical applications such as speech recognition or music genre classification. However, further improvements can be made by using more advanced models or incorporating more data augmentation techniques.
