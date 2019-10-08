from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import random

#session variables
seshname = "apples"

batchrangestart = 1
batchrangeend = 100

epochrangestart = 5
epochrangeend = 20

hiddenstart = 2
hiddenend = 100

optimizer = ["adam","Nadam"]
losses = ["CategoricalCrossentropy"]

resultsfile = open(r"C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/AI/BotStorage/" + seshname + ".txt","w+")

#######################################################
#DATA
#######################################################



#Input data files
data = open("C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/Covertype1train.csv")
data2 = open("C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/Covertype1traincat.csv")
data3 = open("C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/Covertype1test.csv")
data4 = open("C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/Covertype1testcat.csv")

#save state locations
savelocation = "C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/AI/BotStorage"

#Into pandas
train = pd.read_csv(data)
trainc = pd.read_csv(data2)
test = pd.read_csv(data3)
testc = pd.read_csv(data4)

#into NumPy array
train1 = np.array(train)
trainc2 = np.array(trainc)
test3 = np.array(test)
testc4 = np.array(testc)

#into Tensors
data5=tf.convert_to_tensor(train1)
data6=tf.convert_to_tensor(trainc2)
data7=tf.convert_to_tensor(test3)
data8=tf.convert_to_tensor(testc4)
tf.keras.backend.flatten(data5)
tf.keras.backend.flatten(data6)
tf.keras.backend.flatten(data7)
tf.keras.backend.flatten(data8)

#######################################################
#MODEL
#######################################################
report = ""
for iter in range(0,1):
    random.seed(time.perf_counter())
    iterbatch = random.randrange(batchrangestart,batchrangeend)
    iterepoch =random.randrange(epochrangestart,epochrangeend)
    iterhidden = random.randrange(hiddenstart,hiddenend)
    random.seed(time.perf_counter())
    #type of model
    model = tf.keras.Sequential([

    #layers Dense if fully connected
    layers.Dense(44, activation='relu'),
    layers.Dense(iterhidden, activation='relu'),
    layers.Dense(3, activation='softmax')])

    # training var.
    model.compile(optimizer="adam",
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=[tf.keras.metrics.CategoricalAccuracy()])

    #fit the data
    model.fit(data7,data8,epochs=iterepoch,batch_size=iterbatch)
    #save
    botname = seshname + " " + str(time.perf_counter())

    results = model.evaluate(data5,data6)

    report += botname + " batch =" + str(iterbatch) + " epoch =" + str(iterepoch) + " hidden =" + str(iterhidden) + ' test loss, test acc:' + str(results) + "\n"
    newpath = "./AI/BotStorage/" + botname
    os.mkdir(newpath)
    tf.saved_model.save(model,newpath)

resultsfile.write(report)