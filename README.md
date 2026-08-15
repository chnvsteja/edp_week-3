# Week-3 Spam Classifier using Manchine Learning

A Machine Learning project that automatically classifies SMS messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and a supervised machine learning approach.

The project takes raw SMS text as input, performs text preprocessing and cleaning, converts the text into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**, and then uses a **Multinomial Naive Bayes** classifier to predict whether a message is spam or legitimate.

The complete workflow was implemented and tested using **Python in Anaconda/Jupyter Notebook**, followed by model evaluation, custom message testing, and saving the trained model and TF-IDF vectorizer for future use.

---

## 📌 Project Overview

Spam messages are unwanted messages that are commonly sent in bulk for advertising, scams, fraudulent activities, phishing attempts, or other unwanted communication.

Manually identifying spam messages becomes difficult when the number of messages increases. Machine Learning can be used to automatically learn patterns from previously labeled messages and classify new messages based on their content.

This project builds a text classification system that learns from a dataset containing two types of SMS messages:

* **Ham** → Normal or legitimate message
* **Spam** → Unwanted or suspicious message

The system follows a complete Machine Learning pipeline:

```text
Raw SMS Dataset
      ↓
Data Inspection
      ↓
Data Cleaning
      ↓
Text Preprocessing
      ↓
Label Encoding
      ↓
Train-Test Split
      ↓
TF-IDF Feature Extraction
      ↓
Model Training
      ↓
Prediction
      ↓
Model Evaluation
      ↓
Custom SMS Testing
      ↓
Model Saving
```

---

# 🎯 Objectives

The main objectives of this project are:

1. To understand the fundamentals of text classification.
2. To preprocess and clean raw SMS text data.
3. To convert text data into numerical features.
4. To implement TF-IDF feature extraction.
5. To train a supervised Machine Learning classification model.
6. To classify SMS messages into Spam and Ham categories.
7. To evaluate the performance of the trained model.
8. To analyze predictions using a confusion matrix and classification metrics.
9. To test the model using custom SMS messages.
10. To save the trained model and vectorizer for future prediction.
11. To understand a complete end-to-end NLP Machine Learning workflow.

---

# 🧠 Problem Statement

With the increasing number of SMS messages received by users, unwanted spam messages have become a common problem.

Spam messages may contain:

* Promotional content
* Fake offers
* Prize notifications
* Suspicious links
* Financial scams
* Unwanted advertisements
* Fraudulent requests
* Misleading information

The goal of this project is to develop a Machine Learning model that can automatically analyze the content of an SMS message and determine whether it is:

```text
SPAM
```

or

```text
HAM
```

The system should be able to learn patterns from previously labeled messages and use those patterns to classify unseen messages.

---

# 💡 Proposed Solution

The proposed solution uses a Natural Language Processing pipeline combined with a Machine Learning classification algorithm.

The dataset is first loaded and inspected to understand its structure. The SMS messages are then cleaned by converting text to lowercase and removing unnecessary characters.

The target labels are converted into numerical values:

```text
ham  → 0
spam → 1
```

The cleaned messages are divided into training and testing datasets.

Since Machine Learning algorithms cannot directly work with raw text, **TF-IDF Vectorization** is used to transform the SMS messages into numerical feature vectors.

The numerical representations are then provided to a **Multinomial Naive Bayes** classifier.

The trained model predicts the class of test messages, and its performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

Finally, custom SMS messages are passed to the trained pipeline to verify how the classifier performs on unseen examples.

---

# 🏗️ Project Architecture

The overall architecture of the project can be represented as follows:

```text
                 ┌───────────────────────┐
                 │      SMS Dataset      │
                 │      spam.csv         │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Data Inspection     │
                 │  Shape / Columns /    │
                 │ Missing Values /      │
                 │ Duplicate Records     │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    Data Cleaning      │
                 │ Lowercase Conversion  │
                 │ Regex Text Cleaning   │
                 │ Duplicate Removal     │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    Label Encoding     │
                 │                       │
                 │ Ham  → 0              │
                 │ Spam → 1              │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    Train/Test Split   │
                 │       80 / 20         │
                 │  Stratified Sampling  │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    TF-IDF Vectorizer  │
                 │                       │
                 │ Text → Numerical      │
                 │ Feature Representation│
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Multinomial Naive     │
                 │ Bayes Classifier      │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │       Prediction      │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │     Evaluation        │
                 │ Accuracy              │
                 │ Precision             │
                 │ Recall                │
                 │ F1-Score              │
                 │ Confusion Matrix      │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Custom SMS Testing  │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │     Model Saving      │
                 │ spam_model.pkl        │
                 │ tfidf_vectorizer.pkl  │
                 └───────────────────────┘
```

---

# 📂 Dataset

The project uses an SMS spam dataset stored as:

```text
spam.csv
```

The dataset contains SMS messages along with their corresponding classifications.

The original dataset columns were:

```text
v1
v2
```

These were renamed during preprocessing to more meaningful names:

```text
label
message
```

The resulting structure is conceptually:

| label | message                                |
| ----- | -------------------------------------- |
| ham   | Hey, how are you doing today?          |
| spam  | Congratulations! You have won a prize! |
| ham   | Can you call me later?                 |
| spam  | Claim your free reward now!            |

---

# 📊 Dataset Features

The main columns used in the project are:

### `label`

Represents the category of the SMS.

Possible values:

```text
ham
spam
```

### `message`

Contains the actual SMS text.

Example:

```text
"Congratulations! You have won a free prize."
```

The `message` column acts as the primary input feature for the NLP model.

---

# 🔍 Step 1 — Loading the Dataset

The dataset is loaded into a Pandas DataFrame.

The initial objective is to understand:

* Number of records
* Number of columns
* Column names
* Data types
* Missing values
* Duplicate records
* Distribution of labels

Typical inspection operations include:

```python
df.head()
df.shape
df.info()
df.isnull().sum()
df['label'].value_counts()
```

This initial inspection helps understand the quality and structure of the dataset before applying preprocessing.

---

# 🔎 Step 2 — Data Inspection

Before training the Machine Learning model, the dataset is examined for possible problems.

The following aspects were checked:

### Dataset Shape

The shape of the dataset was checked to determine the number of rows and columns.

### Column Information

The dataset structure was inspected to identify the relevant input and target columns.

### Missing Values

Missing values were checked to make sure that important SMS messages or labels were not unavailable.

### Duplicate Records

Duplicate SMS records were identified and removed to avoid unnecessary repetition in the training data.

### Class Distribution

The number of Spam and Ham messages was examined to understand the distribution of the target classes.

---

# 🧹 Step 3 — Data Cleaning

Raw text data often contains unnecessary characters and inconsistent formatting.

Therefore, text preprocessing was performed before feature extraction.

The cleaning process includes:

1. Converting text to lowercase.
2. Removing unnecessary characters.
3. Cleaning special characters using regular expressions.
4. Removing duplicate records.

For example:

### Before Cleaning

```text
"Congratulations!!! YOU have won $1000. Claim NOW!!!"
```

### After Cleaning

```text
"congratulations you have won claim now"
```

The purpose of cleaning is to reduce unnecessary variation in the text while retaining meaningful information.

---

# 🔤 Step 4 — Lowercase Conversion

All SMS messages are converted into lowercase.

For example:

```text
"FREE MONEY NOW"
```

becomes:

```text
"free money now"
```

This prevents the model from treating uppercase and lowercase versions of the same word as completely different features.

---

# 🧽 Step 5 — Regular Expression Text Cleaning

Regular expressions are used to remove unwanted characters from SMS messages.

The preprocessing focuses on retaining meaningful textual information while removing unnecessary symbols and noise.

This helps create a cleaner representation of the SMS messages before TF-IDF vectorization.

---

# 🏷️ Step 6 — Label Encoding

Machine Learning algorithms require numerical target values.

Therefore, the text labels are converted into binary numerical values.

```text
ham  → 0
spam → 1
```

This allows the classifier to learn the relationship between the SMS text and its corresponding category.

The resulting target variable contains:

```text
0 = Ham
1 = Spam
```

---

# ✂️ Step 7 — Train-Test Split

After preprocessing, the dataset is divided into two subsets:

```text
80% → Training Data
20% → Testing Data
```

The training data is used to teach the model.

The testing data is kept separate and is used to evaluate how well the model performs on unseen messages.

A stratified split is used so that the proportion of Spam and Ham messages remains reasonably consistent between the training and testing sets.

The split uses:

```python
test_size=0.2
random_state=42
stratify=y
```

Using `random_state=42` makes the split reproducible.

---

# 🧮 Step 8 — TF-IDF Feature Extraction

Machine Learning models cannot directly understand text.

For example, the model cannot directly process:

```text
"Congratulations you won a free prize"
```

Therefore, the text must be converted into numerical features.

The project uses:

```text
TF-IDF
```

which stands for:

**Term Frequency–Inverse Document Frequency**

---

# 📚 What is TF-IDF?

TF-IDF is a numerical representation technique commonly used in Natural Language Processing.

It determines how important a word is within a document relative to the entire collection of documents.

TF-IDF consists of two major concepts:

### Term Frequency — TF

Measures how frequently a word occurs within a particular document.

### Inverse Document Frequency — IDF

Measures how uncommon a word is across the entire collection of documents.

A simplified representation is:

```text
TF-IDF = Term Frequency × Inverse Document Frequency
```

Words that are common across many messages receive lower importance, while words that are more informative receive higher weights.

---

# ⚙️ TF-IDF Implementation

The project uses:

```python
TfidfVectorizer(stop_words='english')
```

The vectorizer is fitted only on the training messages:

```python
X_train_tfidf = vectorizer.fit_transform(X_train)
```

The test messages are transformed using the same fitted vectorizer:

```python
X_test_tfidf = vectorizer.transform(X_test)
```

This ensures that the feature representation remains consistent between training and testing.

---

# 🤖 Step 9 — Machine Learning Model

The project uses:

# Multinomial Naive Bayes

Multinomial Naive Bayes is a popular classification algorithm for text-based Machine Learning problems.

It is particularly useful for applications involving:

* Text classification
* Spam detection
* Document classification
* Sentiment analysis
* News classification

---

# 🧠 Why Multinomial Naive Bayes?

Multinomial Naive Bayes works well with features that represent word counts or word importance.

TF-IDF produces numerical representations of text that are suitable for this type of classifier.

The algorithm is also:

* Fast
* Lightweight
* Easy to train
* Suitable for high-dimensional text data
* Effective for many text classification tasks

---

# 🔄 Model Training Process

The training process can be summarized as:

```text
Training SMS Messages
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Numerical Feature Matrix
        ↓
Multinomial Naive Bayes
        ↓
Trained Classification Model
```

The model learns patterns that differentiate Spam messages from Ham messages.

For example, words or phrases frequently associated with spam may contribute to the model's prediction.

---

# 🔮 Step 10 — Prediction

After training, the model is used to predict the labels of the test messages.

The prediction process is:

```text
New SMS
   ↓
Text Cleaning
   ↓
TF-IDF Transformation
   ↓
Trained Naive Bayes Model
   ↓
Prediction
   ↓
Spam / Ham
```

The model produces a numerical prediction:

```text
0 → Ham
1 → Spam
```

These numerical predictions can then be converted back into readable labels.

---

# 📈 Step 11 — Model Evaluation

After training the model, its performance is evaluated using the testing dataset.

The project evaluates the classifier using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

These metrics provide a better understanding of model performance than accuracy alone.

---

# 🎯 Accuracy

Accuracy measures the percentage of total predictions that were correct.

The general formula is:

```text
Accuracy =
Correct Predictions / Total Predictions
```

A higher accuracy indicates that the classifier correctly predicts a larger proportion of the test messages.

However, accuracy alone may not fully describe spam detection performance, which is why additional metrics are used.

---

# 🎯 Precision

Precision answers the question:

> Of all messages predicted as Spam, how many were actually Spam?

Formula:

```text
Precision =
True Positives / (True Positives + False Positives)
```

High precision means the model does not incorrectly classify many legitimate messages as spam.

---

# 🎯 Recall

Recall answers:

> Of all actual Spam messages, how many did the model correctly identify?

Formula:

```text
Recall =
True Positives / (True Positives + False Negatives)
```

High recall is important in spam detection because missing a spam message can be undesirable.

---

# 🎯 F1-Score

The F1-score combines Precision and Recall.

Formula:

```text
F1 Score =
2 × (Precision × Recall) /
(Precision + Recall)
```

The F1-score provides a balanced measure when both precision and recall are important.

---

# 📋 Classification Report

The classification report provides a detailed performance summary for each class.

It includes:

```text
precision
recall
f1-score
support
```

for:

```text
0 → Ham
1 → Spam
```

This makes it easier to understand how the model performs separately for legitimate and spam messages.

---

# 🔲 Confusion Matrix

A confusion matrix provides a visual and numerical representation of the classification results.

For binary classification, the confusion matrix contains:

```text
                 Predicted
              Ham       Spam

Actual Ham     TN        FP

Actual Spam    FN        TP
```

Where:

### True Negative — TN

A Ham message correctly classified as Ham.

### True Positive — TP

A Spam message correctly classified as Spam.

### False Positive — FP

A Ham message incorrectly classified as Spam.

### False Negative — FN

A Spam message incorrectly classified as Ham.

The confusion matrix is especially useful for understanding the types of mistakes made by the classifier.

---

# 🧪 Step 12 — Custom SMS Testing

After evaluating the model on the test dataset, custom SMS messages were tested.

This verifies whether the trained pipeline can process completely new messages.

Examples of messages that can be tested include:

```text
"Congratulations! You have won a free prize. Claim now!"
```

Expected type:

```text
Spam
```

Another example:

```text
"Hey, are we meeting for lunch today?"
```

Expected type:

```text
Ham
```

The custom testing process demonstrates how the model can be used outside the original dataset.

---

# 📊 Prediction Probability

The trained Naive Bayes classifier can also provide prediction probabilities.

This can give an indication of how strongly the model associates a message with each class.

Conceptually:

```text
Message:
"Congratulations! Claim your free reward now!"

Ham Probability:
0.02

Spam Probability:
0.98
```

The exact probability depends on the trained model and input message.

This provides additional information beyond simply returning Spam or Ham.

---

# 💾 Step 13 — Saving the Model

After successfully training the classifier, the trained model is saved so that it can be reused without retraining from the beginning.

The trained model is saved as:

```text
models/spam_model.pkl
```

The TF-IDF vectorizer is saved separately as:

```text
models/tfidf_vectorizer.pkl
```

These files allow the prediction system to reuse the trained Machine Learning pipeline later.

---

# 📦 Why Save the Vectorizer?

Saving only the classifier is not sufficient.

The model expects input data in the same numerical feature representation that was used during training.

Therefore, the TF-IDF vectorizer must also be saved.

The prediction workflow becomes:

```text
New SMS
   ↓
Saved TF-IDF Vectorizer
   ↓
Numerical Features
   ↓
Saved Spam Model
   ↓
Prediction
```

This ensures consistency between training and future prediction.

---

# 🗂️ Project Structure

A suitable structure for the project is:

```text
Spam-Classifier/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── spam_classifier.ipynb

```

---

# 📁 Folder Description

## `data/`

Contains the SMS dataset used for training and testing.

```text
data/spam.csv
```

---

## `models/`

Contains the serialized Machine Learning components.

```text
spam_model.pkl
tfidf_vectorizer.pkl
```

---

## `notebooks/`

Contains the Jupyter Notebook used for data analysis, preprocessing, training, evaluation, and testing.

Example:

```text
spam_classifier.ipynb
```

---

## `README.md`

Contains the documentation and explanation of the project.


---

# 🔬 Complete Machine Learning Workflow

The complete implementation can be summarized as follows:

## Phase 1 — Data Collection

```text
spam.csv
```

is used as the source dataset.

---

## Phase 2 — Data Understanding

The dataset is inspected using Pandas.

Important checks include:

```text
Shape
Columns
Data Types
Missing Values
Duplicates
Class Distribution
```

---

## Phase 3 — Data Cleaning

SMS messages are cleaned by:

```text
Lowercase Conversion
        +
Regex Cleaning
        +
Duplicate Removal
```

---

## Phase 4 — Target Encoding

The target labels are converted:

```text
ham  → 0
spam → 1
```

---

## Phase 5 — Dataset Splitting

The dataset is split using:

```text
80% Training
20% Testing
```

with stratification and:

```text
random_state = 42
```

---

## Phase 6 — Feature Engineering

TF-IDF is used to transform text into numerical feature vectors.

```text
Text
 ↓
TF-IDF
 ↓
Numerical Matrix
```

---

## Phase 7 — Model Training

The transformed training data is provided to:

```text
Multinomial Naive Bayes
```

---

## Phase 8 — Prediction

The trained model predicts labels for unseen test messages.

```text
0 → Ham
1 → Spam
```

---

## Phase 9 — Evaluation

The model is evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
Classification Report
Confusion Matrix
```

---

## Phase 10 — Custom Testing

New SMS messages are manually entered and classified.

---

## Phase 11 — Model Persistence

The trained components are saved:

```text
spam_model.pkl
tfidf_vectorizer.pkl
```


# 📝 Project Workflow Summary

The entire project can be summarized in the following sequence:

```text
1. Load spam.csv
        ↓
2. Inspect dataset
        ↓
3. Rename columns
        ↓
4. Check missing values
        ↓
5. Check duplicate records
        ↓
6. Remove duplicates
        ↓
7. Clean SMS text
        ↓
8. Convert text to lowercase
        ↓
9. Remove unwanted characters
        ↓
10. Encode labels
        ↓
11. Split data into train/test
        ↓
12. Apply TF-IDF
        ↓
13. Train Multinomial Naive Bayes
        ↓
14. Generate predictions
        ↓
15. Calculate evaluation metrics
        ↓
16. Generate confusion matrix
        ↓
17. Test custom SMS messages
        ↓
18. Save trained model
        ↓
19. Save TF-IDF vectorizer
```

---

# 🏁 Conclusion

The **Spam Classifier Using Machine Learning** project demonstrates how Natural Language Processing and supervised Machine Learning can be combined to automatically identify unwanted SMS messages.

The project starts with raw SMS data and performs data inspection, cleaning, preprocessing, label encoding, feature extraction, model training, evaluation, and custom prediction.

**TF-IDF** is used to convert textual messages into meaningful numerical features, while **Multinomial Naive Bayes** is used as the classification algorithm.

The model is evaluated using multiple performance metrics including accuracy, precision, recall, F1-score, classification report, and confusion matrix.

The trained model and TF-IDF vectorizer are also saved as `.pkl` files, making the project ready for further development and possible deployment.

Overall, this project provides practical experience with the complete Machine Learning workflow for a real-world Natural Language Processing problem.

---

# 🚀 Quick Start

For a quick setup:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>

cd Spam-Classifier

pip install -r requirements.txt

jupyter notebook
```

Then open:

```text
notebooks/spam_classifier.ipynb
```

and run the notebook from beginning to end.

---

# 👨‍💻 Author

**Chinta N V S Teja**

**Roll Number:** 24951A12B6

---



## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub and exploring the implementation in the Jupyter Notebook.

**Machine Learning → NLP → TF-IDF → Naive Bayes → Spam Detection**
