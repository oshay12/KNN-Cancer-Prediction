# isort was ran
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sbn
import sklearn
import sklearn.model_selection
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# path variable to get file
path = Path(__file__).resolve()

# loading the .mat file into a dictionary
numberFile = loadmat((path.parent / "NumberRecognitionAssignment1.mat").resolve())


def numberRecogData(matFile):
    # grabbing only the training data for eights and nines and putting them
    # into a new dictionary, adding labels
    eightsNinesTraining = {
        "imageArrayTraining8": np.array(list(matFile["imageArrayTraining8"])),
        "imageArrayTraining9": np.array(list(matFile["imageArrayTraining9"])),
        "labels": [0] * len(matFile["imageArrayTraining8"][0][0])
        + [1] * len(matFile["imageArrayTraining9"][0][0]),
    }  # each image of 8 is labeled with an "0" and each 9 labeled with a "1"
    # so that they are easily distingusihable

    # same as above but for testing data, no labels
    eightsNinesTesting = {
        "imageArrayTesting8": np.array(list(matFile["imageArrayTesting8"])),
        "imageArrayTesting9": np.array(list(matFile["imageArrayTesting9"])),
    }

    # testing labels for AUC calculations
    testLabels = {
        "labels": [0] * len(matFile["imageArrayTesting8"][0][0])
        + [1] * len(matFile["imageArrayTesting9"][0][0])
    }

    # reshaping the dimensions of the training arrays to
    # (750, 28, 28) for training and (250, 28, 28), then flattening
    # the images into 2D arrays of shape (750, 784) and (250, 784)
    eightsTrain = (
        eightsNinesTraining["imageArrayTraining8"].transpose(2, 0, 1).reshape(750, 784)
    )

    ninesTrain = (
        eightsNinesTraining["imageArrayTraining9"].transpose(2, 0, 1).reshape(750, 784)
    )

    eightsTest = (
        eightsNinesTesting["imageArrayTesting8"].transpose(2, 0, 1).reshape(250, 784)
    )

    ninesTest = (
        eightsNinesTesting["imageArrayTesting9"].transpose(2, 0, 1).reshape(250, 784)
    )

    # array stacked vertically so images stay separate,
    # but are all combined into one 2D array for knn.fit() and .predict()
    # idea found from:
    # https://www.geeksforgeeks.org/how-to-concatenate-two-2-dimensional-numpy-arrays/
    # under the vstack() portion
    eightsNinesTrain = np.vstack((eightsTrain, ninesTrain))
    eightsNinesTest = np.vstack((eightsTest, ninesTest))

    # training and testing data combined
    trainTest = np.vstack((eightsNinesTrain, eightsNinesTest))

    # originally, I just used the testing portion of 8's and 9's from the .mat
    # file, but I was getting 100% prediction accuracy for almost every k, so I
    # assumed it was because the numbers weren't shuffled around, so I split the
    # training and testing data into train-test splits and shuffled them,
    # this gave more realistic results.
    # idea from the scikit-learn documentation:
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    (
        numTrain,
        numTest,
        numLabelTrain,
        numLabelTest,
    ) = sklearn.model_selection.train_test_split(
        trainTest,
        eightsNinesTraining["labels"] + testLabels["labels"],
        random_state=40,  # random state hard set so analysis is reproducable
        shuffle=True,
        test_size=0.2,
    )  # 80-20 train-test data split

    return numTrain, numTest, numLabelTrain, numLabelTest


def question1(file):
    numTrain, numTest, numLabelTrain, numLabelTest = numberRecogData(file)
    errorRate = []  # array of error rates for tests

    # running the knn model for neighbor values of 1-20
    for i in range(1, 21):
        knn = KNeighborsClassifier(i)
        knn.fit(numTrain, numLabelTrain)
        prediction = knn.predict(numTest)
        auc = roc_auc_score(y_true=numLabelTest, y_score=prediction)
        errorRate.append((1 - auc) * 100)

    def save(errors) -> None:
        arr = np.array(errors)
        if len(arr.shape) > 2 or (len(arr.shape) == 2 and 1 not in arr.shape):
            raise ValueError(
                "Invalid output shape. Output should be an array "
                "that can be unambiguously raveled/squeezed."
            )
        if arr.dtype not in [np.float64, np.float32, np.float16]:
            raise ValueError("Your error rates must be stored as float values.")
        arr = arr.ravel()
        if len(arr) != 20 or (arr[0] >= arr[-1]):
            raise ValueError(
                "There should be 20 error values, with the first value "
                "corresponding to k=1, and the last to k=20."
            )
        if arr[-1] >= 2.0:
            raise ValueError(
                "Final array value too large. You have done something "
                "very wrong (probably relating to standardizing)."
            )
        if arr[-1] < 0.8:
            raise ValueError(
                "You probably have not converted your error rates to percent values."
            )
        outfile = Path(__file__).resolve().parent / "errors.npy"
        np.save(outfile, arr, allow_pickle=False)
        print(f"Error rates succesfully saved to {outfile }")

    save(errorRate)

    # plotting aesthetics
    sbn.set_style(style="darkgrid")
    plt.xlabel("K values")
    plt.ylabel("Testing Error Rate")
    plt.title("Testing Error Rate of KNN Predictions (Number Recognition) vs. K values")
    # to specify the tick spacing on the graph using locators
    # idea from chatgpt
    x_locator = MultipleLocator(base=1.0)
    y_locator = MultipleLocator(base=0.25)
    plt.gca().xaxis.set_major_locator(x_locator)
    plt.gca().yaxis.set_major_locator(y_locator)
    # adding percent symbol to y values
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.plot(
        range(1, 21),
        errorRate,
        color="blue",
        marker="o",
        linewidth=1,
        markersize=5,
    )
    plt.savefig("knn_q1.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()


# loading csv into dataframe for q2 and q3 using pandas
breastCancerDF = pandas.read_csv((path.parent / "breast-cancer.csv.xls").resolve())


# helper fuction for accessing the breast cancer dataframe's splits
def breastCancerData(df):
    # cleaning up data
    df = df.drop(["id"], axis=1)  # don't need id values

    # representing the labels numerically for the model
    df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})
    # splitting df into just the features and just the labels
    testingData = df.drop(["diagnosis"], axis=1)
    labels = df["diagnosis"]

    # normalizing data in between 0 and 1
    # using sklearn's MinMaxScaler. adapted from:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    # looping through dataframe to normalize each column
    for i in testingData:
        testingData[i] = MinMaxScaler().fit_transform(
            testingData[i].values.reshape(-1, 1)
        )

    # preparing the data into train-test splits as they are not provided
    (
        bcTrain,
        bcTest,
        bcLabelTrain,
        bcLabelTest,
    ) = sklearn.model_selection.train_test_split(
        testingData, labels, random_state=40, shuffle=True, test_size=0.2
    )  # 80-20 train-test split

    return testingData, labels, bcTrain, bcTest, bcLabelTrain, bcLabelTest


def question2(bcFile):
    # grabbing data from helper function
    testingData, labels = breastCancerData(bcFile)[0], breastCancerData(bcFile)[1]
    # testingData, labels = breastCancerData()[0]
    # initializing array to hold AUC values
    aucValues = []

    # using the index-locator function to grab columns at certain indicies
    # from the dataframe, comparing row values within that column to
    # the assigned group, malignant or benign, and getting AUC from that
    for i in range(len(testingData.columns)):
        auc = sklearn.metrics.roc_auc_score(
            y_true=labels, y_score=testingData.iloc[:, i]
        )
        # label and auc score assigned, held in arrays and
        # rounded to 3 decimal places
        feature = [testingData.columns[i], auc.round(3)]
        aucValues.append(feature)

    # sort auc values in descending order
    aucValues = sorted(aucValues, key=lambda x: x[1], reverse=True)

    # column labels for graphing
    labelsArray = [label[0] for label in aucValues]

    # AUC values for graphing
    aucArray = [feature[1] for feature in aucValues]

    # reset to default so no crossover if multiple functions are ran
    plt.rcdefaults()
    plt.xlabel("AUC Scores")
    plt.tick_params(axis="y", which="major", labelsize=10)
    sbn.barplot(y=labelsArray[0:10], x=aucArray[0:10], orient="horizontal").set(
        title="Ten Most Important Features shown through AUC values"
    )
    plt.xlim(0.75, 1.0)
    plt.xticks(np.arange(0.75, 1.0, 0.05))  # make AUC values tick up by 0.05
    plt.gca().set_axisbelow(True)  # make gridlines appear under bars
    plt.grid()
    plt.tight_layout()  # so the full feature names are shown
    plt.savefig("bonus1.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()


def question3(bcFile):
    # grabbing splits from helper function
    bcTrain, bcTest, bcLabelTrain, bcLabelTest = (
        breastCancerData(bcFile)[2],
        breastCancerData(bcFile)[3],
        breastCancerData(bcFile)[4],
        breastCancerData(bcFile)[5],
    )
    errorRate = []  # error rates array

    # running the knn model for neighbor values of 1-20
    for i in range(1, 21):
        knn = KNeighborsClassifier(i)
        knn.fit(bcTrain.values, bcLabelTrain.values)
        prediction = knn.predict(bcTest.values)
        auc = roc_auc_score(y_true=bcLabelTest, y_score=prediction)
        errorRate.append((1 - auc) * 100)

    # plotting aesthetics
    sbn.set_style(style="darkgrid")
    plt.xlabel("K values")
    plt.ylabel("Testing Error Rate")
    plt.title("Testing Error Rate of KNN Predictions (Cancer Prediciton) vs. K values")
    # to specify the tick spacing on the graph using locators
    # idea from chatgpt
    x_locator = MultipleLocator(base=1.0)
    y_locator = MultipleLocator(base=0.25)
    plt.gca().xaxis.set_major_locator(x_locator)
    plt.gca().yaxis.set_major_locator(y_locator)
    # adding percent symbol to y values
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.plot(
        range(1, 21),
        errorRate,
        color="blue",
        marker="o",
        linewidth=1,
        markersize=5,
    )
    plt.tight_layout()
    plt.savefig("knn_q3.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()


def bonus(numFile, bcFile):
    # I was curious of the accuracy, precision, recall and F1 scores,
    # as well as the confusion matrix for these tests. so I
    # found methods to obtain them from the sklearn.metrics documentation at:
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    # looading number and breast cancer data
    numTrain, numTest, numLabelTrain, numLabelTest = numberRecogData(numFile)
    testingData, labels, bcTrain, bcTest, bcLabelTrain, bcLabelTest = breastCancerData(
        bcFile
    )
    # K = 5 seemed like a good predictor for number recognition, not 100%
    # in predicting but very close, so I will use K=5
    numKNN = KNeighborsClassifier(5)
    numKNN.fit(numTrain, numLabelTrain)
    numKNNPredictions = numKNN.predict(numTest)
    # for the breast cancer dataset, I found K=7 to be the best predictor,
    # so I will be using that
    cancerKNN = KNeighborsClassifier(7)
    cancerKNN.fit(bcTrain.values, bcLabelTrain.values)
    cancerKNNPredictions = cancerKNN.predict(bcTest.values)

    # % of correct predictions over total number of samples
    numAccuracy = accuracy_score(numLabelTest, numKNNPredictions).round(3)
    cancerAccuracy = accuracy_score(bcLabelTest, cancerKNNPredictions).round(3)

    # % of true positives classified correctly
    numRecall = recall_score(numLabelTest, numKNNPredictions).round(3)
    cancerRecall = recall_score(bcLabelTest, cancerKNNPredictions).round(3)

    # % of correct predictions over true positive and true negative predictions
    numPrecision = precision_score(numLabelTest, numKNNPredictions).round(3)
    cancerPrecision = precision_score(bcLabelTest, cancerKNNPredictions).round(3)

    # harmonic mean of precision and recall
    numF1 = f1_score(numLabelTest, numKNNPredictions).round(3)
    cancerF1 = f1_score(bcLabelTest, cancerKNNPredictions).round(3)

    # confusion matrix:
    # (topleft=trueNeg, botleft=falseNeg, topright=falsePos, botright=truePos)
    numCFMatrix = confusion_matrix(numLabelTest, numKNNPredictions)
    cancerCFMatrix = confusion_matrix(bcLabelTest, cancerKNNPredictions)

    print(
        "\nNUMBER RECOGNITION",
        "\nValues all for 5 neighbor KNN model:",
        "\n------------------------------------------",
        "\nAccuracy: ",
        numAccuracy,
        "\nRecall: ",
        numRecall,
        "\nPrecision: ",
        numPrecision,
        "\nF1 Score: ",
        numF1,
        "\nConfusion Matrix: \n",
        numCFMatrix,
        "\n------------------------------------------",
    )

    print(
        "\nBREAST CANCER PREDICTIONS",
        "\nValues all for 7 neighbor KNN model:",
        "\n------------------------------------------",
        "\nAccuracy: ",
        cancerAccuracy,
        "\nRecall: ",
        cancerRecall,
        "\nPrecision: ",
        cancerPrecision,
        "\nF1 Score: ",
        cancerF1,
        "\nConfusion Matrix: \n",
        cancerCFMatrix,
        "\n------------------------------------------",
    )

    # was also curious if I would get better (or comparable) prediction
    # accuracy and a more lightweight model using only the top ten features
    # from AUC calculations in question 2 for KNN.

    # stolen code from question 2 to get the top ten
    aucValues = []

    for i in range(len(testingData.columns)):
        auc = sklearn.metrics.roc_auc_score(
            y_true=labels, y_score=testingData.iloc[:, i]
        )
        feature = [testingData.columns[i], auc.round(3)]
        aucValues.append(feature)

    aucValues = sorted(aucValues, key=lambda x: x[1], reverse=True)
    labelsArray = [label[0] for label in aucValues]

    # drop every feature that isn't in the top ten
    topTenFeatures = testingData.drop(
        columns=testingData.columns.difference(labelsArray[0:10])
    )

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        topTenFeatures, labels, random_state=40, shuffle=True, test_size=0.2
    )

    errorRate = []  # error rates array

    # running the knn model for neighbor values of 1-20
    for i in range(1, 21):
        knn = KNeighborsClassifier(i)
        knn.fit(x_train, y_train)
        prediction = knn.predict(x_test)
        auc = roc_auc_score(y_true=y_test, y_score=prediction)
        errorRate.append((1 - auc) * 100)

    # K=3 gave the best results for this data,
    # so I will use that for accuracy, precision, recall and F1 scores,
    # and the confusion matrix
    topTenKNN = KNeighborsClassifier(3)
    topTenKNN.fit(x_train, y_train)
    ttKNNPredictions = topTenKNN.predict(x_test)

    ttAccuracy = accuracy_score(y_test, ttKNNPredictions).round(3)
    ttPrecision = precision_score(y_test, ttKNNPredictions).round(3)
    ttRecall = recall_score(y_test, ttKNNPredictions).round(3)
    ttF1 = f1_score(y_test, ttKNNPredictions).round(3)
    ttCFMatrix = confusion_matrix(y_test, ttKNNPredictions)

    print(
        "\nBREAST CANCER PREDICTIONS (Top Ten Features)",
        "\nValues all for 3 neighbor KNN model:",
        "\n------------------------------------------",
        "\nAccuracy: ",
        ttAccuracy,
        "\nRecall: ",
        ttRecall,
        "\nPrecision: ",
        ttPrecision,
        "\nF1 Score: ",
        ttF1,
        "\nConfusion Matrix: \n",
        ttCFMatrix,
        "\n------------------------------------------",
    )

    # plotting aesthetics
    sbn.set_style(style="darkgrid")
    plt.xlabel("K values")
    plt.ylabel("Testing Error Rate")
    plt.title("Testing Error Rate of KNN Predictions (Top Ten Features) vs. K values")
    # to specify the tick spacing on the graph using locators
    # idea from chatgpt
    x_locator = MultipleLocator(base=1.0)
    y_locator = MultipleLocator(base=0.25)
    plt.gca().xaxis.set_major_locator(x_locator)
    plt.gca().yaxis.set_major_locator(y_locator)
    # adding percent symbol to y values
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.plot(
        range(1, 21),
        errorRate,
        color="blue",
        marker="o",
        linewidth=1,
        markersize=5,
    )
    plt.tight_layout()
    plt.savefig("bonus2.png")
    # clear figure so no aesthetics get carried over to future plots
    plt.clf()


def main():
    question1(numberFile)
    question2(breastCancerDF)
    question3(breastCancerDF)
    bonus(numberFile, breastCancerDF)


if __name__ == "__main__":
    main()
