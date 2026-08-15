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
