# Model Prediction Streamlit App

This repository contains a Streamlit application that allows users to interact with four pre-trained machine learning models for prediction tasks: CustomCNN, ResNet50, EfficientNet-B2, and VGG16. The app loads .pth model files and provides a user interface to test the models.

## Team Members

- _Name:_ Md. Samir Hossain _ID:_ 2022-3-60-161
- _Name:_ Rukaiah Bintay Morshed _ID:_ 2022-3-60-334

## Installation Instructions

1. _Download Files from GitHub:_  
   Clone this repository or download app.py and requirements.txt from the GitHub repository and place them in a local folder.

2. _Create and Activate a Virtual Environment:_  
   In the local folder, create a virtual environment and activate it:

   python -m venv venv

   venv\Scripts\activate

3. _Download Model Files from Google Drive:_  
   Go to the following Google Drive link: [https://drive.google.com/drive/folders/1sx0k4mZHHgPslZla67TbtffHHPeLSpBd?usp=drive_link](https://drive.google.com/drive/folders/1sx0k4mZHHgPslZla67TbtffHHPeLSpBd?usp=drive_link).  
   Download the file named pth files.zip.  
   Unzip the archive and copy all the .pth files into the same local folder where app.py and requirements.txt are located.

4. _Install Dependencies:_  
   Open a terminal in the folder and run:

   pip install -r requirements.txt

5. _Run the App:_  
   Execute the following command to start the Streamlit app:

   streamlit run app.py

## Models

The app supports the following four models:

- _CustomCNN_: A custom convolutional neural network model.
  [https://www.kaggle.com/code/samirhossain2001/mulberry-customcnn](https://www.kaggle.com/code/samirhossain2001/mulberry-customcnn)
- _ResNet50_: A deep residual network with 50 layers.
  [https://www.kaggle.com/code/samirhossain2001/mulberry-resnet50](https://www.kaggle.com/code/samirhossain2001/mulberry-resnet50)
- _EfficientNet-B2_: An efficient convolutional neural network optimized for performance.
  [https://www.kaggle.com/code/samirhossain2001/mulberry-efficientnet-b2](https://www.kaggle.com/code/samirhossain2001/mulberry-efficientnet-b2)
- _VGG16_: A convolutional neural network with 16 layers known for its simplicity and depth.
  [https://www.kaggle.com/code/samirhossain2001/mulberry-vgg16](https://www.kaggle.com/code/samirhossain2001/mulberry-vgg16)

## Authors

- [@SamirHossain2001](https://github.com/SamirHossain2001)
