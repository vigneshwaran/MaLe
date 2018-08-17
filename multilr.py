from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data=datasets.load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=["MEDV"])
lm=linear_model.LinearRegression()
lm.fit(df,target)
p=lm.predict(df)
#plt.scatter(df,target)
#plt.plot(df,p)
#plt.show()

x_surf, y_surf = np.meshgrid(np.linspace(df.AGE.min(), df.AGE.max(), 30),np.linspace(df.TAX.min(), df.TAX.max(), 30))

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x_surf, y_surf,c='blue', marker='o')
ax.plot_surface(x_surf, y_surf,p.reshape(x_surf))