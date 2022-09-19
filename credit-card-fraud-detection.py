import pandas as dataframe
from sklearn.model_selection import train_test_split as skSelectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as accuracyMetrics

# Declaring function to get accuracy of transaction
def findAccuracy(predictionData,comparingData,type):
    dataAccuracy = accuracyMetrics(predictionData, comparingData)
    print("Accuracy on ",type," data : ", dataAccuracy)

# Reading available PCA transformation information inside
# credit card csv and loading to dataframe
creditCardDataset = dataframe.read_csv('content/creditcard.csv')
# Taking first and last 10 records from dataset
topTenRows = creditCardDataset.head(10)
print("================ First Ten Row========================== ")
print(topTenRows)

lastTenRow = creditCardDataset.tail(10)
print("================ Last Ten Row========================== ")
print(lastTenRow)

# Checking datatype, available columns, non-null values and how much memory usage
creditCardDataset.info()

# In our dataset we have 30 columns Time, V1, V2...V28, Amount, Class
# so it will check any missing value available in each column
checkNullValue = creditCardDataset.isnull().sum()
print("========================= Checking for missing value ==================== ")
print(checkNullValue)



# Give number of valid transaction and fraud transaction in csv
print("============= Count for total number of valid and fraud transaction =====================")
totalTransaction = creditCardDataset['Class'].value_counts()
print("Total valid transaction : ", totalTransaction.loc[0]);
print("Total fraud transaction : ", totalTransaction.loc[1]);


# Current data in csv is highly unbalanced, so
# it need to separate transaction based on value in Class column in csv
# if class column having value 0 than it will consider as Valid Transaction
# if class column having value 1 than it will consider as Fraud Transaction
print("============= Valid transaction data start =====================")
validTransaction = creditCardDataset[creditCardDataset.Class== 0]
print(validTransaction.shape)
print("============= Valid transaction data end =====================")

print("============= Fraud transaction data start =====================")
fraudTransaction = creditCardDataset[creditCardDataset.Class== 1]
print(fraudTransaction.shape)
print("============= Fraud transaction data end =====================")


# get description for Amount in dataset for both valid and fraud transaction such as
# count, mean(average), standard deviation etc.
validTransactionStatistics = validTransaction.Amount.describe()
fraudTransactionStatistics = fraudTransaction.Amount.describe()
print("================ Valid Transaction Statistics ====================")
print(validTransactionStatistics)
print("================ Fraud Transaction Statistics ====================")
print(fraudTransactionStatistics)

# Comparing both valid and fraud transaction and get average from that
creditCardDataset.groupby('Class').mean()

# Merging both valid transaction(random 52 record) and fraud transaction data set
# based on index
mergedDataset = dataframe.concat([validTransaction.sample(n = 52), fraudTransaction], axis=0)
firstTenMergedData = mergedDataset.head(10)
print("================ First Ten Row Of Merged Dataset ========================== ")
print(firstTenMergedData)



lastTenMergedData = mergedDataset.tail(10)
print("================ last Ten Row Of Merged Dataset ========================== ")
print(lastTenMergedData)





mergedDataset['Class'].value_counts()
compareBothTransaction = mergedDataset.groupby('Class').mean()
print("============ Comparison Between Both Transaction On Merged Dataset =====")
print(compareBothTransaction)



# Drop column Class from dataset and split dataframe for feature and target data
X = mergedDataset.drop(columns='Class', axis =1 )
Y = mergedDataset['Class']
print("================ Data in X ======================== ")
print(X)
print("================ Data in Y ======================== ")
print(Y)

# Splitting data in two
# 1) Training Data
# 2) Testing Data
X_training, X_testing, Y_training, Y_testing = skSelectors(X, Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_training.shape, X_testing.shape)

# Logistic Regression model
logisticRegressionModel = LogisticRegression()
# training the Logistic Regression Model with Training Data
logisticRegressionModel.fit(X_training, Y_training)
# accuracy on training data
XtrainingDataPrediction = logisticRegressionModel.predict(X_training)
findAccuracy(XtrainingDataPrediction,Y_training,"Training")

# accuracy on testing data
XtestingDataPrediction = logisticRegressionModel.predict(X_testing)
findAccuracy(XtestingDataPrediction,Y_testing,"Testing")

