import numpy as np
import matplotlib.pyplot as plt

# NumPy Exercise
wt, ht = np.random.random(100) * 50 + 40, np.random.random(100) * 0.6 + 1.4
bmi = wt / (ht * ht)
print(bmi[:10])

# MatPlotLib Exercise
status = (bmi >= 18.5).astype(int) + (bmi >= 25.0) + (bmi >= 30.0)
# status = 0: underweight, 1: health, 2: overweight, 3: obese
sstr = ['underweight', 'healthy', 'overweight', 'obese']

# Boxplot
def plotBox(a, s):
    plotData = []
    for i in range(4): plotData.append(a[status==i])
    plt.boxplot(plotData)
    plt.xticks(np.arange(4)+1, sstr)
    plt.xlabel('BMI Category')
    plt.ylabel(s)
    plt.show()

plotBox(ht, 'Height (m)')
plotBox(wt, 'Weight (kg)')

# Histogram (1)
plt.hist(status, bins=range(5), rwidth=.8, align='left')
plt.xticks(range(4), sstr)
plt.xlabel('BMI category')
plt.ylabel('Number of students')
plt.show()

# Histogram (2)
def plotHistogram(a, s):
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.hist(a[status==i], rwidth=.8)
        plt.title(sstr[i])
        plt.xlabel(s)
        plt.ylabel('Number of students')
    plt.tight_layout()
    plt.show()

plotHistogram(ht, 'Height (m)')
plotHistogram(wt, 'Weight (kg)')

# Pie chart
plotData = []
for i in range(4): plotData.append(bmi[status==i].size)
plt.pie(plotData, labels=sstr, autopct='%1.2f%%')
plt.show()

# Scatter plot
for i in range(4):
    plt.scatter(ht[status==i], wt[status==i], label=sstr[i])
plt.xlabel('Height (m)')
plt.ylabel('Weight (kg)')
plt.legend(loc='upper right')
plt.show()
