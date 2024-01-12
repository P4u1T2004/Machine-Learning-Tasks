##linearregressionfromscratch
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')

def lossfunction(m,b,points):
    totalerror=0
    for i in range(len(points)):
        x=points.iloc[i].studytime
        y=points.iloc[i].score
        totalerror+=(y - (m*x + b))**2
    totalerror/float(len(points))

def gradientdescent(mnow,bnow,points,L):
    mgrad=0
    bgrad=0
    n=len(points)

    for i in range(len(points)):
        x=points.iloc[i].studytime
        y=points.iloc[i].score
        
        mgrad=(-2/n)*x*(y-(mnow*x+bnow))
        bgrad=(-2/n)*(y-(mnow*x+bnow))

    m=mnow-mgrad*L
    b=bnow-bgrad*L
    return m,b
m=0
b=0
L=0.0001
epochs=1000
for i in range(epochs):
    m,b=gradientdescent(m,b,data,L)

print(m,b)
plt.scatter(data.studytime,data.score,color="black")
plt.plot(list(range(1,100)),[m*x+b for x in range(1,100)],color="red")
plt.show()
