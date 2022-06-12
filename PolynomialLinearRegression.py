import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Took the dataset with pandas and imported it into the dataframe object df.
df = pd.read_csv("SalaryData.csv",sep = ";")

df

plt.scatter(df['experience'],df['salary'])
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.savefig('1.png', dpi=300)
plt.show()

# The data are not distributed in a linear structure.
# If linear regression is applied to this dataset, there will be an unsuitable prediction line:

reg = LinearRegression()
reg.fit(df[['experience']],df['salary'])

plt.xlabel('Experience (years)')
plt.ylabel('Salary')

plt.scatter(df['experience'],df['salary'])   

xlabel = df['experience']
ylabel = reg.predict(df[['experience']])
plt.plot(xlabel, ylabel,color= "green", label = "linear regression")
plt.legend()
plt.show()

# Call the PolynomialFeatures function to create a polynomial regression object.
# Specify the degree (N) of the polynomial:
polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(df[['experience']])


# Fit the x polynomial and y axes by creating the reg object and calling its fit method.
# Train the regression model with real data:
reg = LinearRegression()
reg.fit(x_polynomial,df['salary'])

# The model is ready and trained, now let's see how the model creates a result graph according to the data:

y_head = reg.predict(x_polynomial)
plt.plot(df['experience'],y_head,color= "red",label = "polynomial regression")
#plt.plot(xaxis, yaxis,color= "green", label = "linear regression")
plt.legend()

# Scatter the dataset with dots.
plt.scatter(df['experience'],df['salary'])   

plt.show()

# Now let's make N=3 or 4 and see if we increase the polynomial degree, will it fit better?

x_polynomial1 = polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial1)