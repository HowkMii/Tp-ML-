import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("training.csv")
target = data["FraudResult"]
y =data.iloc[:,-1]


data = data.drop(["TransactionId","SubscriptionId" ,"CustomerId","ProductCategory","CurrencyCode","ProviderId","ProductId","ChannelId","CountryCode","TransactionStartTime"], axis = 1)





Xafterdrop = data.iloc[:,:].values
Xnew = data.iloc[:,:].values
y = target.iloc[:].values

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Xnew[:,0] = label.fit_transform(Xnew[:,0])
Xnew[:,1] = label.fit_transform(Xnew[:,1])
Xnew[:,-1] = label.fit_transform(Xnew[:,-1])
y = label.fit_transform(y)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xnew = scaler.fit_transform(Xnew)


from sklearn.decomposition import PCA
pca = PCA(2)
red_X = pca.fit_transform(Xnew)
ratio = pca.explained_variance_ratio_
y = label.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(red_X,y , test_size = 0.2 , random_state = 0)

############# Logistic Regression
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

print('Start.............')

classifier = LogisticRegression()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print('The score  ' , r2_score(y_test , y_pred))
print('The accuracy  ', (y_pred == y_test).mean())
cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix is :', cm)
print('######################################')
