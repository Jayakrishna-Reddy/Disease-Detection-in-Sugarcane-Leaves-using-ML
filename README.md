# Disease Detection in Sugarcane Leaves using Machine Learning

## Objective
- [x] Data Acquisition:
Image Dataset: The dataset consists of images of sugarcane leaves in .jpg or .jpeg format. These images represent different conditions:
Healthy leaf
Red Rot
Red Rust
The images are labeled based on the condition of the leaf (healthy, Red Rot, Red Rust). This data is used for training and testing the model.

- [x] Preprocessing:
Image Preprocessing: Since the dataset consists of images, some preprocessing steps would likely be needed to make the images suitable for CNNs. This could include:
Resizing the images to a uniform size.
Normalizing pixel values (scaling them to a range like [0,1]).
Augmenting the dataset (using techniques like rotation, zoom, or flipping) to improve the model's generalization.

- [x] Transfer Learning for Feature Extraction:
Transfer learning uses a pre-trained CNN model to extract relevant features from the images. Here, the two pre-trained CNN models used are:

DenseNet201: A deep CNN model with a dense architecture that facilitates the reuse of features across layers. DenseNet has shown good performance on image classification tasks.

VGG16: A simpler CNN architecture, which consists of 16 layers, including convolutional layers and fully connected layers. It has been successful for many image classification tasks, especially for transfer learning.

Fine-tuning: Both DenseNet201 and VGG16 are used in their pre-trained form (i.e., weights initialized from training on large datasets like ImageNet). Only the forward propagation is used for feature extraction. This means that the convolutional layers will be used to extract features, but the weights of the pre-trained layers are not updated during training. Instead, only the classifier layer (i.e., output layer) is trained.

- [x] Classification Models:
Once the features are extracted using DenseNet201 or VGG16, traditional machine learning algorithms are used for the final classification. These classifiers are:

Support Vector Machine (SVM): A supervised learning algorithm that finds the optimal hyperplane that separates classes in the feature space. It is particularly effective in high-dimensional spaces and has been widely used for image classification tasks.
K-Nearest Neighbors (KNN): A simple, yet powerful classification algorithm that assigns a class based on the majority class among its k-nearest neighbors in the feature space.
Random Forest: An ensemble learning method that combines multiple decision trees to improve classification accuracy and reduce overfitting. It aggregates predictions from multiple trees to produce a final classification.

- [x] Prediction:
The final classification is based on the output of one of these classifiers (SVM, KNN, or Random Forest). The system will predict the disease condition of a sugarcane leaf as one of the following:
Healthy
Red Rot
Red Rust

- [x] Model Evaluation:
After training the models, you would evaluate the performance of the classifiers using metrics such as:

Accuracy: The percentage of correct predictions made by the model.
Precision, Recall, and F1-Score: These are important in classifying imbalanced datasets where one class may have fewer samples than others.
Confusion Matrix: This helps in visualizing how well the model distinguishes between the disease classes (Red Rot, Red Rust) and healthy leaves.

- [x] Outcome:
The final system provides an automated solution for detecting diseases in sugarcane leaves. By analyzing the image and extracting features with pre-trained CNN models, followed by classification with SVM, KNN, or Random Forest, the system can predict whether a leaf is healthy or infected with Red Rot or Red Rust.


- [x] Conclusion:
In summary, this research has successfully demonstrated the use of transfer learning with DenseNet201 and VGG16 models for predicting diseases in sugarcane leaves. The core of the approach involved leveraging pre-trained models, fine-tuning them to adapt to the specific problem of disease detection, and extracting key features from the images in the dataset. These features were then passed into the classification models—SVM, KNN, and Random Forest—which were employed to predict the health status of the sugarcane leaves, distinguishing between healthy, Red Rot, and Red Rust conditions.

The comparison between the classifiers—SVM, KNN, and Random Forest—highlighted variations in their performance, with each algorithm exhibiting distinct strengths and weaknesses in terms of accuracy, classification reports, and training time. These differences reflect the unique characteristics of each algorithm and their behavior when applied to the dataset.

An important takeaway from this research is the significant influence of dataset quality on the choice of machine learning model. The performance of the classifiers was shown to vary depending on the dataset used, which indicates that there is no one-size-fits-all approach to classification tasks. Rather, the choice of algorithm should be informed by an understanding of the dataset’s characteristics, as well as the specific requirements of the classification task at hand. This underscores the importance of conducting thorough experiments and analyses to choose the most suitable machine learning model for a given application.

In conclusion, while transfer learning with DenseNet201 and VGG16 proved effective in extracting meaningful features, the choice of classification algorithm must be tailored to the dataset and its inherent properties. Future work can explore further optimization techniques and additional models to improve disease detection accuracy in agricultural applications.