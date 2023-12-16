#!/usr/bin/env python
# coding: utf-8

# # LJ Rusell
# 
# ## Programming 2 Data Infrastructure project
# 
# ### Who uses Linkedin?

# In[12]:


#Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

import pandas as pd

# Construct the full file path
Social_Media = '/Users/lionelrussell/Downloads/social_media_usage1.csv'

# Read in the data
s = pd.read_csv(Social_Media)

# Check the dimensions of the DataFrame
print("Dimensions of the DataFrame:", s.shape)


# In[13]:


#Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

import numpy as np
import pandas as pd

# Define the clean_sm function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Create a toy DataFrame
toy_df = pd.DataFrame({
    'Column1': [1, 2, 0],  # Sample data
    'Column2': [0, 1, 3]
})

# Apply the function to the DataFrame
toy_df = toy_df.applymap(clean_sm)

# Display the modified DataFrame
print(toy_df)


# In[14]:


#Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.
ss = pd.DataFrame()
ss['sm_li'] = s['web1h'].apply(lambda x: 1 if x == 1 else 0) 
ss['income'] = s['income'].apply(lambda x: x if 1 <= x <= 9 else np.nan)
ss['education'] = s['educ2'].apply(lambda x: x if 1 <= x <= 8 else np.nan)
ss['age'] = s['age'].apply(lambda x: x if x <= 98 else np.nan)

ss['parent'] = s['par'].apply(lambda x: 1 if x == 1 else 0)
ss['married'] = s['marital'].apply(lambda x: 1 if x == 1 else 0)
ss['female'] = s['gender'].apply(lambda x: 1 if x == 2 else 0)


# In[18]:


#exploration: Age vs Linkedin usage
import seaborn as sns
import matplotlib.pyplot as plt

# Setting the style for the plot
sns.set(style="whitegrid")

# Creating a box plot to explore the relationship between age and LinkedIn usage
plt.figure(figsize=(8, 6))
sns.boxplot(data=ss, x='sm_li', y='age')
plt.title('Relationship between Age and LinkedIn Usage')
plt.xlabel('LinkedIn Usage (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()


# In[19]:


#Exploration of Education vs linkedin usage:

import seaborn as sns
import matplotlib.pyplot as plt

# Setting the style for the plot
sns.set(style="whitegrid")

# Creating a bar plot to explore the relationship between education level and LinkedIn usage
plt.figure(figsize=(10, 6))
sns.countplot(data=ss, x='education', hue='sm_li')
plt.title('Relationship between Education Level and LinkedIn Usage')
plt.xlabel('Education Level')
plt.ylabel('Count of LinkedIn Users/Non-Users')
plt.legend(title='LinkedIn Usage', labels=['No', 'Yes'])
plt.show()


# In[20]:


#Exploration of Marriage Status vs Linkedin Usage.

import seaborn as sns
import matplotlib.pyplot as plt

# Setting the style for the plots
sns.set(style="whitegrid")

# Creating a plot for marital status vs LinkedIn usage
plt.figure(figsize=(8, 6))
sns.countplot(data=ss, x='married', hue='sm_li')
plt.title('Impact of Marital Status on LinkedIn Usage')
plt.xlabel('Marital Status (0 = Not Married, 1 = Married)')
plt.ylabel('Count')
plt.legend(title='LinkedIn Usage', labels=['Non-User', 'User'])
plt.show()


# In[21]:


#Exploration of Gender vs Linkedin Usage

import seaborn as sns
import matplotlib.pyplot as plt

# Setting the style for the plots
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(data=ss, x='female', hue='sm_li')
plt.title('Impact of Gender on LinkedIn Usage')
plt.xlabel('Gender (0 = Male/Other, 1 = Female)')
plt.ylabel('Count')
plt.legend(title='LinkedIn Usage', labels=['Non-User', 'User'])
plt.show()


# In[15]:


# Drop all rows with missing values
ss_clean = ss.dropna()


# In[10]:


#Create a target vector (y) and feature set (X)

# Target vector (y) - Contains values of the variable to predict
y = ss['sm_li']

# Feature set (X) - Contains predictor variables
X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]


# In[22]:


#Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning
from sklearn.model_selection import train_test_split
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[24]:


#Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced')
X_train_dropped = X_train.dropna()
y_train_dropped = y_train[X_train_dropped.index]  # Ensure that y_train aligns with X_train

# Fit the model with the non-missing data
logreg.fit(X_train_dropped, y_train_dropped)


# In[29]:


#Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform it
X_train_imputed = imputer.fit_transform(X_train)

# Transform the test data with the fitted imputer
X_test_imputed = imputer.transform(X_test)

# Fit the model with the imputed training data
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train_imputed, y_train)  # Use the original y_train, no need to drop because we imputed X_train

# Evaluate accuracy on the imputed test set
accuracy = logreg.score(X_test_imputed, y_test)
print(f"Accuracy of the model: {accuracy:.2f}")

# Use the model to make predictions on the imputed test set
y_pred = logreg.predict(X_test_imputed)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# #Model Interpretation
# 
# In evaluating the performance of my logistic regression model using a confusion matrix, I observed several key aspects. Firstly, the model identified 132 instances correctly where individuals were not using LinkedIn; these are known as true negatives. This high number suggests that my model is quite effective at recognizing individuals who aren't LinkedIn users.
# 
# However, I also noticed that there were 66 instances classified as false positives. In these cases, my model mistakenly predicted that certain individuals were using LinkedIn when, in fact, they were not. This relatively high number of false positives compared to the 81 true positives (where the model correctly identified LinkedIn users) indicates a tendency of my model to mislabel non-users as users.
# 
# On the positive side, the model had 22 instances of false negatives, where it incorrectly predicted that individuals were not using LinkedIn when they actually were. The lower count of false negatives is encouraging as it means the model is not missing out on many actual LinkedIn users.
# 
# Overall, my interpretation of these results suggests that while my model is better at identifying non-users of LinkedIn, it also has a noticeable inclination to falsely classify non-users as users. Moving forward, I'll need to consider this in refining the model, especially in contexts where misidentifying non-users as users could have significant implications.

# In[30]:


#Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

import pandas as pd
from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Convert the confusion matrix to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, 
                              columns=['Predicted Not Using LinkedIn', 'Predicted Using LinkedIn'], 
                              index=['Actual Not Using LinkedIn', 'Actual Using LinkedIn'])

print(conf_matrix_df)


# In[31]:


# Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.
TP = 81
FP = 66
FN = 22

# Calculations
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")


# - **Precision**: Think of precision as being really careful about not making a mistake. For instance, when your email app tries to filter out spam, you really don't want it to mistakenly put important emails in the spam folder. So, in this case, we focus on precision to ensure we're only calling an email 'spam' if we're really sure about it.
# 
# - **Recall**: Recall is about not missing any important instances. Imagine a doctor screening for a serious disease; it's really important not to miss any case. So, we focus on recall to try to catch every single actual case, even if it means we get some false alarms.
# 
# - **F1 Score**: The F1 score helps when you want a good balance between being careful (precision) and not missing cases (recall), especially when the situation isn't evenly balanced (like if one outcome is a lot more common than the other). It's like trying to find a middle ground that considers both not making mistakes and not missing any important instances.

# In[32]:


#After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

from sklearn.metrics import classification_report

# Generate the classification report
report = classification_report(y_test, y_pred)
print(report)


# My hand calculations for precision, recall, and the F1 score match exactly with the classification report for the class labeled '1' (which I assume represents LinkedIn users in your model). This confirms that your manual calculations are correct.
# 
# The classification report also provides metrics for class '0' (non-users), as well as macro and weighted averages across both classes. The macro average gives equal weight to each class, while the weighted average gives more weight to the more frequent class. My model seems to perform better in terms of recall for class '1' (LinkedIn users), which is crucial in scenarios where missing out on actual users (false negatives) can be more critical.
# 
# Overall, the matching metrics indicate that my manual calculations are accurate and align with the automated metrics provided by scikit-learn's classification_report.

# In[34]:


#Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?
import numpy as np




