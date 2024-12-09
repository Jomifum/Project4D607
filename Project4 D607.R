---
  title: "Project4D607 Document classification"
author: "Jose Fuentes"
date: "2024-12-07"
output: html_document
---
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
##Introduction

In today's digital age, email communication is ubiquitous, yet it brings with it the persistent challenge of spam. Spam emails not only clutter our inboxes but also pose significant security risks. This project aims to develop a robust spam classification system using a dataset from the SpamAssassin Public Corpus. By leveraging machine learning techniques, specifically the Naive Bayes classifier, we can analyze and predict whether an email is spam or ham (non-spam). This predictive model is trained on a labeled dataset and then tested on new, unseen emails to evaluate its accuracy and effectiveness. The ultimate goal is to enhance email filtering mechanisms, ensuring a cleaner and safer email experience.

##Process

This project began by acquiring a labeled dataset of spam and ham emails from the SpamAssassin Public Corpus. The data was imported into R, creating a structured dataset with email content and corresponding labels. To prepare the data for analysis, the text was cleaned by converting it to lowercase, removing punctuation, numbers, stop words, and extra spaces, and handling non-ASCII characters. The text data was transformed into a numerical format suitable for machine learning by creating a Document-Term Matrix (DTM).

To evaluate model performance, the data was split into training and testing sets. A Naive Bayes classifier was trained on the training data and then tested on the unseen test data, demonstrating high accuracy and reliable performance metrics. We further validated the model by applying it to new emails to predict their spam or ham classification. To gain insights into the most frequent words in the dataset, this was visualized them by using a word cloud. Each stage of this process, from data preparation to model training and evaluation, contributed to the development of a robust spam classification system.

```{r project}

library(tm)        # For text mining and preprocessing
library(e1071)     # For Naive Bayes classifier
library(caret)     # For model evaluation
library(wordcloud) # For data visualization
library(RColorBrewer) # For color palettes in word clouds

# Set paths for spam and ham folders
spam_dir <- "C:/Users/Dell/Downloads/Project4D607/spam_2"
ham_dir <- "C:/Users/Dell/Downloads/Project4D607/easy_ham"

# Function to load emails
load_emails <- function(directory) {
  files <- list.files(directory, full.names = TRUE)
  sapply(files, function(x) paste(readLines(x, warn = FALSE, encoding = "UTF-8"), collapse = " "))
}

# Load spam and ham emails
spam_emails <- load_emails(spam_dir)
ham_emails <- load_emails(ham_dir)

# Combine emails and create labels
emails <- c(spam_emails, ham_emails)
labels <- factor(c(rep("spam", length(spam_emails)), rep("ham", length(ham_emails))))

# Preprocess the data
# Create a Corpus
corpus <- Corpus(VectorSource(emails))

# Function to remove non-ASCII characters
removeNonASCII <- content_transformer(function(x) iconv(x, "latin1", "ASCII", sub=""))

# Text Preprocessing
corpus <- tm_map(corpus, removeNonASCII)                   # Remove non-ASCII characters
corpus <- tm_map(corpus, content_transformer(tolower))     # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)                # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)                    # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("english"))# Remove stopwords
corpus <- tm_map(corpus, stripWhitespace)                  # Strip extra whitespace

# Create Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99)  # Keep only terms appearing in 1% of documents

# Split data into training and testing sets
set.seed(123)
train_indices <- createDataPartition(labels, p = 0.8, list = FALSE)
train_dtm <- dtm[train_indices, ]
test_dtm <- dtm[-train_indices, ]
train_labels <- labels[train_indices]
test_labels <- labels[-train_indices]

# Train a Naive Bayes Model
# Convert to matrix for Naive Bayes
train_matrix <- as.matrix(train_dtm)
test_matrix <- as.matrix(test_dtm)

# Train the model
nb_model <- naiveBayes(train_matrix, train_labels)

# Evaluate the model
predictions <- predict(nb_model, test_matrix)

# Confusion matrix and performance metrics
confusion_matrix <- confusionMatrix(predictions, test_labels)
print(confusion_matrix)

# Predict new emails
new_emails <- c(
  "Congratulations! You've won a free prize! Click here to claim.",
  "Hey, can you send me the files for the project by tonight?"
)

new_corpus <- Corpus(VectorSource(new_emails))

# Preprocess new emails
new_corpus <- tm_map(new_corpus, removeNonASCII)            # Remove non-ASCII characters
new_corpus <- tm_map(new_corpus, content_transformer(tolower))
new_corpus <- tm_map(new_corpus, removePunctuation)
new_corpus <- tm_map(new_corpus, removeNumbers)
new_corpus <- tm_map(new_corpus, removeWords, stopwords("english"))
new_corpus <- tm_map(new_corpus, stripWhitespace)

# Create DTM for new emails
new_dtm <- DocumentTermMatrix(new_corpus, control = list(dictionary = Terms(dtm)))
new_matrix <- as.matrix(new_dtm)

# Predict
new_predictions <- predict(nb_model, new_matrix)
print("Predictions for new emails:")
print(new_predictions)  # Output: "spam" or "ham"

# Visualize most common words in the dataset (optional)
wordcloud(corpus, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2"))

```

#Conclusion

This project successfully demonstrated the application of a Naive Bayes classifier to distinguish between spam and ham emails using the SpamAssassin Public Corpus. With a high accuracy rate and reliable performance metrics, the model effectively identified spam emails. The preprocessing steps, including text cleaning and feature extraction, were crucial in preparing the data for analysis. While the model performed well, continuous improvements, such as incorporating additional features and exploring other machine learning algorithms, can further enhance its accuracy and robustness. This project underscores the importance of using machine learning techniques in real-world applications to improve digital communication security and efficiency.

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
