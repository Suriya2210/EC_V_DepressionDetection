import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read in the data and preprocess it
depression_data = pd.read_csv('D:\Depression.csv')
depression_data.loc[depression_data['class'] == 'suicide', 'class',] = 1
depression_data.loc[depression_data['class'] == 'non-suicide', 'class',] = 0

X = depression_data['text']
Y = depression_data['class']

# Test and Train Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()

model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Accuracy on training data : ',accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

def predict(input_entry, result_label):
    text = input_entry.get()
    input_mail = [text]
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)

    if prediction[0] == 0:
        result_label.config(text='Result: Non-depressed 👍')
    else:
        result_label.config(text='Result: Depressed 😔')

# Set up the Tkinter GUI
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title('Depression Prediction')
        self.geometry('1920x1080')

        # Add a label and an input field for the user to enter text
        self.input_label = tk.Label(self, text='Enter text:', font=('Montserrat', 18))
        self.input_entry = tk.Entry(self, font=('Montserrat', 18), width=50)

        # Add a button to initiate the prediction
        self.predict_button = tk.Button(self, text='Predict', font=('Montserrat', 18), command=lambda: predict(self.input_entry, self.result_label))

        # Add a label to display the prediction result
        self.result_label = tk.Label(self, text='', font=('Montserrat', 18))

        # Add the widgets to the GUI
        self.input_label.pack(padx=25,pady=25)
        self.input_entry.pack(padx=25,pady=25)
        self.predict_button.pack(padx=25,pady=25)
        self.result_label.pack(padx=25,pady=25)

# Run the GUI
app = App()
app.mainloop()
