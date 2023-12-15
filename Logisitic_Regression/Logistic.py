import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

dataset = pd.read_csv("Classifier/LogisticReg/Social_Network_Ads.csv")

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values




X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.2,random_state=0)



sc = StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)


reg = LogisticRegression(random_state=0)

reg.fit(X_train,y_train)


y_pred = reg.predict(X_test)

np.set_printoptions(precision=2)


print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

x_set,y_set = sc.inverse_transform(X_train),y_train

x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-10,stop=x_set[:,0].max()+10,step=.25),
                    np.arange(start = x_set[:,1].min()-1000,stop=x_set[:,1].max()+1000,step=.25))

plt.contourf(x1,x2,reg.predict(sc.transform(np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),
            alpha=.75,cmap= ListedColormap(("red","green")))

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(("blue","green"))(i),label=j)


plt.legend()
plt.show()

x_set,y_set = sc.inverse_transform(X_test),y_test

x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-10,stop=x_set[:,0].max()+10,step=.25),
                    np.arange(start = x_set[:,1].min()-1000,stop=x_set[:,1].max()+1000,step=.25))

plt.contourf(x1,x2,reg.predict(sc.transform(np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),
            alpha=.75,cmap= ListedColormap(("red","green")))

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
  plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(("blue","green"))(i),label=j)


plt.legend()
plt.show()

