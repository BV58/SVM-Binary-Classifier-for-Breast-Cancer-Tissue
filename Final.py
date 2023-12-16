import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

#Loading data
df = pd.read_csv('data.csv')

#Dropping id columns in data.csv
df.drop(df.columns[0], axis=1, inplace=True)
for column in df.columns:
    if "Unnamed" in column:
        df.drop(column, axis = 1, inplace=True)

#Processing data
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

#Split the data
#X = Features
#Y = Labels (Malignant/Benign)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initliaziing SVM as our classifier
classifier = SVC(kernel='linear', random_state=42)

#Training the model based on SVM
classifier.fit(X_train, y_train)

#Determining accuracy of SVM Classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#Output
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))


#Generate Coefficient Magnitude graph
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=X.columns, y=classifier.coef_[0], palette="viridis")
plt.title("Feature Importance in SVM Classifier")
plt.xlabel("Features")
plt.ylabel("Coefficient Magnitude")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

