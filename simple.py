from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('Gpu')))
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import random

#before running you will need these libraries
#pip install TensorFlow
#pip install keras
#pip install pandas
#pip install numpy
#and anything else it yells at you for not having when you try and run.
#it will try and use a gpu, but should still chooch if you dont have one, you will just get a bunch of red its cool tho

#session variables
#this is the name that you give for each training iteration. It appears in the results file and a prefix for each net.
seshname = "donut"

#The following are the hyperparameters: the start and endpoints for the rng. These will need to be
#modified over time to compare with other ranges for finding the perfect net.

#batch size- how many it feeds into net before changing weights or something like that
batchrangestart = 10
batchrangeend = 1000

#each epoch is one iteration over the entire training set. Sometimes you need a lot, but rule of thumb is the lower
#you can get this the better. The network starts to memorize the training set instead of learning general patters
#if this is too large. Depends on problem/data
epochrangestart = 3
epochrangeend = 10

# number of nodes in te hidden layer. The network below is only 3 layers input->hidden->output. Very generic
#and simple for a program called simple.py. Different problems will use wildly different structures that
#have been found to be effective.
hiddenstart = 5
hiddenend = 200

#not currently implemented, but idea is to vary the optimizer function and loss function as one, this data set
#really doesnt benefit from this so it isnt very needed atm.
optimizer = ["adam","Nadam"]
losses = ["CategoricalCrossentropy"]

#not currently implemented, idea was to add n many hidden layers
hiddenlstart =1
hiddenlend = 5

#there the results will be stored
resultsfile = r"C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/AI/BotStorage/" + seshname + ".csv"

#######################################################
#DATA
#######################################################


#Input data files, train and test are backwords in name but not content because im smart.
data = open("C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/Covertype1train.csv")
data2 = open("C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/Covertype1traincat.csv")
data3 = open("C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/Covertype1test.csv")
data4 = open("C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/Covertype1testcat.csv")

#save state locations, this dir will be filled with folders one for each net
savelocation = "C:/Users/Luke/PycharmProjects/OPERATIONMAKEMONEY/AI/BotStorage"

#the next three chucks can probably be simplified, but it was early in the learning process and when
#i got it to work I kept it so here it is pandas->numpy->tensor

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

#flattening the data. Looks kinda like this [[1],[1]]->[1,1]
tf.keras.backend.flatten(data5)
tf.keras.backend.flatten(data6)
tf.keras.backend.flatten(data7)
tf.keras.backend.flatten(data8)

#######################################################
#MODEL
#######################################################

#stuff for writing resutls does not need to be modified as long as you have the dir above correct for your comp.
#cant remember if it creates or justs adds, so if it gives you shit just create an empty whatever it wants.
writetofile = open(resultsfile, "w+")
writetofile.write("Name,Batch Size,Epoch,Hidden Layer,Loss,Test Result\n")
writetofile.close()

#of bots that will be trained and saved
for iter in range(0,500):

    #new seed for random
    random.seed(time.perf_counter())
    #randomize hyperparameters using the ranges at the top.

    iterbatch = random.randrange(batchrangestart,batchrangeend)
    iterepoch =random.randrange(epochrangestart,epochrangeend)
    iterhidden = random.randrange(hiddenstart,hiddenend)
    iterlhidden = random.randrange(hiddenlstart, hiddenlend)

    #type of model, sequential is exactly what it sounds like 1->2->3
    model = tf.keras.Sequential([

    #layers Dense means every node a has a connnection to every node above it.
    #input layer is 44 because the shape of data is 44,1
    #relu is an activation function. these are applied to the data at entry, some can make exp data into linear and
    # vise vera, you can shape real time data, you can do a bunch of stuff there.
    layers.Dense(44, activation='relu'),

    #hidden layer, sometimes I get better results(with this data set) without it being there tbh, but it's there because AI
    layers.Dense(iterhidden, activation='relu'),

    #output layer is 7 for 7 categories of data, softmax scales the sum of the outputs to 1, so it like confidence %
    layers.Dense(7, activation='softmax')])

    # makes the net and gives it the functions that will determine how it sees how well its doing.
    model.compile(optimizer="Nadam",
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=[tf.keras.metrics.CategoricalAccuracy()])

    #fit the data
    model.fit(data7,data8,epochs=iterepoch,batch_size=iterbatch)

    #save
    #do the data stuff and make report
    botname = seshname + " " + str(time.perf_counter())
    results = model.evaluate(data5,data6)
    resultsformatted = str(results).replace('[', '')
    resultsformatted = resultsformatted.replace(']', '')
    resultsformatted = resultsformatted.replace(' ', '')


    report = botname + "," + str(iterbatch) + "," + str(iterepoch) + "," + str(iterhidden) + "," + resultsformatted + "," + "\n"
    writetofile = open(resultsfile, "a")
    writetofile.write(report)
    writetofile.close()
    newpath = "./AI/BotStorage/" + botname
    os.mkdir(newpath)
    tf.saved_model.save(model,newpath)


