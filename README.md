# Detecting depression from text

This project is a machine learning model that predicts whether a person is depressed or not based on their written text. The model is trained on a dataset of depression-related texts, and can be used to classify new texts as depressed or non-depressed.

## How it works
The model uses logistic regression and Tfidf vectorization to learn patterns in the training data and make predictions on new text data. The input text is first transformed into a numerical representation using the Tfidf vectorizer, which is then fed into the logistic regression model to produce a prediction.

## Dependencies
- numpy
- pandas
- scikit-learn
- tkinter

## Usage
Enter a piece of text into the input field and click the "Predict" button to get the model's prediction.

## GUI
The GUI is built using Tkinter, and consists of an input field, a button to initiate the prediction, and a label to display the result.

## Note
The model's accuracy on the test data is printed to the console, but this value should only be used as a rough guide to the model's performance. To properly evaluate the model, it is necessary to use more robust methods such as cross-validation and the use of highly relevant datasets.
