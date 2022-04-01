import numpy as np
import csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def processPrescription(data, index, offset, numMonths):
    grouping = []
    for j in range (offset, offset + numMonths):
        grouping.append(int(data[index][j]))
    return grouping

def loadData(entries, data):
    for i in range(1, len(data)):
        entry = Entry(int(data[i][0]))
        entry.setName(data[i][1], data[i][2])
        entry.setState(data[i][3])
        entry.setProduct(data[i][4])

        entry.setNRxMonths(processPrescription(data, i, 5, 6))
        entry.setTRxMonths(processPrescription(data, i, 11, 6))
        entries.append(entry)

def checkEntries(entries):
    totalErrors = 0
    for i in range(0, len(entries)):
        # Get the monthly data for each type of prescription
        NRxData = Entry.getNRxMonths(entries[i])
        TRxData = Entry.getTRxMonths(entries[i])
        # Loop through the prescription months
        for j in range(0, len(NRxData)):
            # We shouldn't have more new prescriptions than total prescriptions
            if(NRxData[j] > TRxData[j]):
                # Uh-oh spaghetio. Give a warning
                # print(f"Error with {entries[i]}. Had {NRxData[j]} new prescriptions, but {TRxData[j]} total prescriptions.")
                totalErrors+=1
    print(f"Encountered {totalErrors} total Errors")


class Entry:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return str(self.id) + ": " + self.fName + ", " + self.lName

    def getId(self):
        return self.id

    def setName(self, fName, lName):
        self.fName = fName
        self.lName = lName

    def getName(self):
        return self.fName, self.lName

    def setState(self, State):
        self.State = State

    def getState(self):
        return self.State

    def setProduct(self, product):
        self.product = product
    
    def getProduct(self):
        return self.product

    def setNRxMonths(self, counts):
        self.NRxMonths = counts

    def getNRxMonths(self):
        return self.NRxMonths
    
    def setTRxMonths(self, counts):
        self.TRxMonths = counts

    def getTRxMonths(self):
        return self.TRxMonths

# Make list for entries
entries = []

# Open file to be parsed
with open("Prescriber_Data.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = list(csv_reader)

# Parse the provided CSV
loadData(entries, data)

# Print out everything for testing purposes.
#for i in range(0, len(entries)):
#    print(entries[i])

checkEntries(entries)

# Rank doctors by total prescription
productList = []
productRank = []
for i in range(0, len(entries)):
    product = Entry.getProduct(entries[i])
    if product not in productList:
        productList.append((product))
        productRank.append([])
    TRxMonths = Entry.getTRxMonths(entries[i])
    sum = TRxMonths[0] + TRxMonths[1] + TRxMonths[2] + TRxMonths[3] + TRxMonths[4] + TRxMonths[5]
    doctorTotalPrescriptions = (sum, Entry.getId(entries[i]))
    productRank[productList.index(product)].append(doctorTotalPrescriptions)

for i in range(0,len(productRank)):
    productRank[i].sort()
    productRank[i].reverse()

for i in range(0, len(productRank)):
    print(productList[i])
    print(productRank[i])

# Predict Prescription Trends
productDataTotalPrescription = []
for i in range(0, len(productRank)):
    productDataTotalPrescription.append([0, 0, 0, 0, 0, 0])

for i in range(0, len(entries)):
    product = Entry.getProduct(entries[i])
    TRxMonths = Entry.getTRxMonths(entries[i])
    productDataTotalPrescription[productList.index(product)][0] +=  TRxMonths[0]
    productDataTotalPrescription[productList.index(product)][1] +=  TRxMonths[1]
    productDataTotalPrescription[productList.index(product)][2] +=  TRxMonths[2]
    productDataTotalPrescription[productList.index(product)][3] +=  TRxMonths[3]
    productDataTotalPrescription[productList.index(product)][4] +=  TRxMonths[4]
    productDataTotalPrescription[productList.index(product)][5] +=  TRxMonths[5]

monthSet1 = [1, 2, 3, 4, 5, 6]
linearSet1 = []
predictSet1 = []
for i in range(0, len(productDataTotalPrescription)):
    x = np.array(list(zip(monthSet1)))
    y = productDataTotalPrescription[i]
    func = LinearRegression().fit(x, y)
    linearSet1.append(func)
    predictSet1.append(func.predict(x))
    print(func.coef_)

for i in range(0, len(predictSet1)):
    plt.plot(monthSet1,predictSet1[i],'-',linewidth=3, label=productList[i])
    plt.plot(monthSet1,productDataTotalPrescription[i],'o',linewidth=3, label=productList[i])
plt.xlabel('Months')
plt.ylabel('Predicted Total Prescriptions')
plt.suptitle('Total Prescription Trends')
plt.grid(True)
plt.legend()
plt.savefig('prediction_models_1.png')
plt.show()

# Predict Future Targets
productDataNewPrescription = []
for i in range(0, len(productRank)):
    productDataNewPrescription.append([0, 0, 0, 0, 0, 0])

for i in range(0, len(entries)):
    product = Entry.getProduct(entries[i])
    NRxMonths = Entry.getNRxMonths(entries[i])
    productDataNewPrescription[productList.index(product)][0] +=  NRxMonths[0]
    productDataNewPrescription[productList.index(product)][1] +=  NRxMonths[1]
    productDataNewPrescription[productList.index(product)][2] +=  NRxMonths[2]
    productDataNewPrescription[productList.index(product)][3] +=  NRxMonths[3]
    productDataNewPrescription[productList.index(product)][4] +=  NRxMonths[4]
    productDataNewPrescription[productList.index(product)][5] +=  NRxMonths[5]

monthSet2 = [1, 2, 3, 4, 5, 6, 12]
linearSet2 = []
predictSet2 = []
for i in range(0, len(productDataTotalPrescription)):
    x = np.array(list(zip(monthSet1)))
    y = productDataNewPrescription[i]
    func = LinearRegression().fit(x, y)
    linearSet2.append(func)
    x_p = np.array(list(zip(monthSet2)))
    predictSet2.append(func.predict(x_p))
    print(func.coef_)

for i in range(0, len(predictSet2)):
    plt.plot(monthSet2,predictSet2[i],'-',linewidth=3, label=productList[i])
    plt.plot(monthSet1,productDataNewPrescription[i],'o',linewidth=3, label=productList[i])
plt.xlabel('Months')
plt.ylabel('Predicted New Prescriptions')
plt.suptitle('New Prescription Trends')
plt.grid(True)
plt.legend()
plt.savefig('prediction_models_2.png')
plt.show()