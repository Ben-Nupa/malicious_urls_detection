import pandas as pd
from datetime import datetime,date
import os

ratioGB = 1 #ratio (Good Data)/(Bad Data)
amountVal = 10000 #Amount of validating data
startingDate = date(2015,12,1)
endingDate = date(2018,11,18)

def createDataset(ratioGB, amountVal, startingDate, endingDate) :
    # ratioGB : float : ratio (Good Data)/(Bad Data)
    # amountVal : int : Number of validating urls
    # startingDate : datetime.date : Starting Date (e.g. : date(2015,12,1))
    # endingDate : datetime.date : Ending Date

    urls = [] #list containing the final urls
    label = [] #list containing the final labels

    #Exctracting bad URLS
    print("Exctracting bad urls...")
    badDataset = os.path.join("datasets", "bad_urls1.csv")
    badDataframe: pd.DataFrame = pd.read_csv(badDataset,";")
    badUrls = badDataframe["URL"].tolist()
    badDates = badDataframe["Date"].tolist()


    for i in range(len(badUrls)) :
         if startingDate < datetime.strptime(str(badDates[i]),'%Y%m%d').date() < endingDate :
             #Appending bad urls to the dataset
             urls.append(badUrls[i])
             label.append("bad")

    #Exctracting good URLS
    print("Exctracting good urls...")
    goodDataset = os.path.join("datasets", "good_urls.csv")
    goodDataframe: pd.DataFrame = pd.read_csv(badDataset,";")
    goodUrls = badDataframe["URL"].sample(int(len(urls)*ratioGB)).tolist()

    for i in range(len(goodUrls)) :
        #Appending good urls to the dataset
        urls.append(badUrls[i])
        label.append("good")

    dataFrame = pd.DataFrame({'URL':urls,'Label':label})
    dataFrame = dataFrame.sample(frac = 1) #shuffling data

    valData = dataFrame.iloc[:amountVal,:]
    trainData = dataFrame.iloc[amountVal:,:]

    print("Created a dataset with "+str(trainData.shape[0])+" training urls and "+str(valData.shape[0])+" validating urls")
    return trainData,valData

