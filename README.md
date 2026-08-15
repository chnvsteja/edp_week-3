# Week-3 Spam Classifier Using Machine Learning

A machine learning project developed as part of **EDP Week 3** to automatically classify SMS messages as **Spam** or **Ham** using Natural Language Processing and the **Multinomial Naive Bayes** algorithm.

The project demonstrates a complete text classification workflow, starting from raw SMS data and ending with predictions on new messages.

---

## About the Project

SMS spam is a common problem where users receive unwanted promotional messages, fake offers, and fraudulent notifications.

In this project, an SMS dataset is processed and transformed into machine-readable numerical features using **TF-IDF**. A **Multinomial Naive Bayes** classifier is then trained to distinguish between legitimate and spam messages.

The trained model is also tested with new SMS messages to demonstrate real-world classification.

---

## Project Goals

The main goals of this project are:

- Understand and prepare a real-world text dataset.
- Perform basic text cleaning and preprocessing.
- Convert categorical spam labels into numerical values.
- Divide the dataset into training and testing sets.
- Extract useful text features using TF-IDF.
- Train a Multinomial Naive Bayes classification model.
- Measure the performance of the trained classifier.
- Visualize classification results using a confusion matrix.
- Test the classifier with custom SMS messages.
- Save the trained model and TF-IDF vectorizer for future use.

---

## Dataset Used

The project uses the **SMS Spam Collection** dataset containing labeled SMS messages.

Each message belongs to one of two categories:

| Label | Meaning |
|---|---|
| `ham` | Normal / legitimate SMS |
| `spam` | Unwanted / spam SMS |

The dataset was converted into a CSV file named:

```text
spam.csv
Dataset Format

The CSV file contains:

v1 → Message category
v2 → SMS message

These columns were renamed during preprocessing to:

label
message
Tools and Libraries

The project was implemented using Anaconda and Jupyter Notebook.

Programming Language
Python
Libraries
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Regular Expressions
Joblib
Machine Learning
TF-IDF Vectorization
Multinomial Naive Bayes
Implementation Process

The project was completed through the following stages:

SMS Dataset
     ↓
Load Dataset
     ↓
Data Inspection
     ↓
Remove Unnecessary Columns
     ↓
Rename Columns
     ↓
Check Missing Values
     ↓
Remove Duplicate Records
     ↓
Text Preprocessing
     ↓
Label Encoding
     ↓
Train/Test Split
     ↓
TF-IDF Transformation
     ↓
Naive Bayes Training
     ↓
Prediction
     ↓
Performance Evaluation
     ↓
Custom SMS Testing
     ↓
Save Model
1. Loading the Dataset

The dataset was loaded into a Pandas DataFrame.

df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

Only the required columns containing the message label and SMS text were selected.

df = df[['v1', 'v2']]


df.columns = ['label', 'message']
2. Dataset Exploration

The dataset was inspected to understand its structure and distribution.

The following operations were performed:

df.shape
df.info()
df['label'].value_counts()

The distribution of spam and ham messages was also visualized using a count plot.

sns.countplot(
    data=df,
    x='label'
)

This helped identify the class distribution before training the model.

3. Data Cleaning

The dataset was checked for missing values:

df.isnull().sum()

Duplicate records were also identified and removed:

df = df.drop_duplicates()

This ensured that duplicate SMS messages would not unnecessarily influence the model.

4. Text Preprocessing

The SMS messages were cleaned before feature extraction.

The preprocessing function performs the following operations:

Converts text to lowercase.
Removes numbers and special characters.
Removes unnecessary spaces.
Removes leading and trailing spaces.
def preprocess_text(text):


    text = text.lower()


    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )


    text = re.sub(
        r'\s+',
        ' ',
        text
    )


    return text.strip()

The cleaned text was stored in a separate column:

df['clean_message'] = df['message'].apply(
    preprocess_text
)
Example

Original:

Congratulations!!! You have WON $1000!!!

Processed:

congratulations you have won
5. Converting Labels

The text labels were converted into numerical values so that they could be used by the machine learning model.

df['label_num'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

Therefore:

0 → Ham
1 → Spam
6. Selecting Features and Target

The cleaned SMS messages were used as the input features.

X = df['clean_message']

The numerical spam/ham labels were used as the target.

y = df['label_num']
7. Training and Testing Data

The dataset was divided into training and testing portions.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

The split used:

80% Training Data
20% Testing Data

The training data is used to learn patterns from SMS messages, while the testing data is used to evaluate the model on unseen messages.

8. TF-IDF Vectorization

Raw text cannot be directly provided to the Naive Bayes algorithm.

Therefore, the cleaned SMS messages were converted into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency).

tfidf = TfidfVectorizer(
    stop_words='english'
)

The vectorizer was fitted on the training data:

X_train_tfidf = tfidf.fit_transform(
    X_train
)

The same vectorizer was then applied to the testing data:

X_test_tfidf = tfidf.transform(
    X_test
)

The TF-IDF representation allows the model to identify words that are useful for distinguishing spam messages from legitimate messages.

9. Training the Naive Bayes Classifier

A Multinomial Naive Bayes classifier was selected for the classification task.

model = MultinomialNB()

The model was trained using the TF-IDF features:

model.fit(
    X_train_tfidf,
    y_train
)

After training, the model learned patterns associated with both spam and legitimate SMS messages.

10. Generating Predictions

Predictions were generated using the test dataset:

y_pred = model.predict(
    X_test_tfidf
)

The predicted values represent:

0 → Ham
1 → Spam
11. Model Evaluation

Several metrics were used to evaluate the classifier.

Accuracy

Measures the overall percentage of correctly classified messages.

accuracy_score(
    y_test,
    y_pred
)
Precision

Measures how many messages predicted as spam were actually spam.

precision_score(
    y_test,
    y_pred
)
Recall

Measures how many actual spam messages were correctly detected.

recall_score(
    y_test,
    y_pred
)
F1 Score

Provides a combined measure of precision and recall.

f1_score(
    y_test,
    y_pred
)

A complete classification report was generated using:

classification_report(
    y_test,
    y_pred,
    target_names=['Ham', 'Spam']
)
12. Confusion Matrix

A confusion matrix was generated to understand the types of predictions made by the classifier.

cm = confusion_matrix(
    y_test,
    y_pred
)

The matrix was visualized using Seaborn:

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Ham', 'Spam'],
    yticklabels=['Ham', 'Spam']
)

The confusion matrix represents:

                 Predicted
              Ham       Spam


Actual Ham     TN         FP


Actual Spam    FN         TP

This makes it easier to identify incorrectly classified SMS messages.

13. Testing Custom SMS Messages

After training, the model was tested with new SMS messages that were not part of the original dataset.

Example messages:

test_messages = [
    "Hey bro, are you coming to college tomorrow?",
    "Congratulations! You won a free iPhone. Click now!",
    "Can you call me when you reach home?",
    "URGENT! You have won 10000 dollars. Claim your prize now!",
    "What time is the class today?"
]

The messages were first processed using the same preprocessing function and then converted into TF-IDF features.

clean_test_messages = [
    preprocess_text(message)
    for message in test_messages
]


test_tfidf = tfidf.transform(
    clean_test_messages
)

The trained model was then used for prediction:

predictions = model.predict(
    test_tfidf
)

The final output classifies each message as:

HAM

or

SPAM
14. Spam Probability

The classifier was also used to obtain prediction probabilities.

probabilities = model.predict_proba(
    test_tfidf
)

This provides an estimate of how strongly the model associates a message with the spam class.

For example:

Message: Congratulations! You won a free iPhone!


Prediction: SPAM
Spam Probability: 98.42%
15. Saving the Trained Model

The trained Naive Bayes model was saved using Joblib.

joblib.dump(
    model,
    "models/spam_model.pkl"
)

The TF-IDF vectorizer was also saved:

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

These files can be reused later to classify new messages without retraining the model.

Project Files
EDP_WK-3/
│
├── spam.csv
│
├── Week3_Spam_Classifier.ipynb
│
├── README.md
│
├── requirements.txt
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── results/
    └── confusion_matrix.png
Results

The model was evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix

The classifier was also tested using custom SMS messages to verify that it could distinguish between normal and spam messages.

The exact evaluation scores are available in the Jupyter Notebook.

Key Learning Outcomes

Through this project, the following concepts were implemented:

Text data cleaning
Natural Language Processing fundamentals
Label encoding
Train/test splitting
TF-IDF feature extraction
Naive Bayes classification
Classification metrics
Confusion matrix analysis
Prediction on unseen text
Saving machine learning models
Future Enhancements

The project can be extended by:

Adding stemming or lemmatization.
Experimenting with different text preprocessing techniques.
Comparing Naive Bayes with Logistic Regression and SVM.
Performing hyperparameter tuning.
Handling class imbalance using suitable techniques.
Building a web interface for real-time SMS classification.
Deploying the trained model as an API.
Conclusion

This project demonstrates a complete machine learning pipeline for SMS spam detection.

Starting with raw SMS data, the messages were cleaned and transformed into numerical representations using TF-IDF. A Multinomial Naive Bayes classifier was then trained to distinguish between legitimate and spam messages.

The model was evaluated using multiple performance metrics and tested with new SMS messages, demonstrating how Natural Language Processing and machine learning can be combined to build a practical spam detection system.

Author

Ch NVS Teja

EDP - Week 3
