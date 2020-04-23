import random
from Network import NeuralNetwork
'''
train the model with NN library made by me !~
Creating data 3 labels : trains, cats, trains.
the data including 3000 images:
2400 ----> training data(800 for every label)
600 ----> testing data(200 for every label)
the data downloaded from google drawit! game.
'''
all_data = []
cats_training = []
trains_training = []
rainbows_training = []
cats_testing = []
trains_testing = []
rainbows_testing = []

with open("cats1000new.bin", "rb") as f:  # open the file
    byte = f.read(1)
    while byte != b"":
        byte = f.read(1)
        all_data.append(byte)

with open("trains1000new.bin", "rb") as f:  # open the file
    byte = f.read(1)
    while byte != b"":
        byte = f.read(1)
        all_data.append(byte)

with open("rainbow1000new.bin", "rb") as f:  # open the file
    byte = f.read(1)
    while byte != b"":
        byte = f.read(1)
        all_data.append(byte)

for i in range(len(all_data)):
    all_data[i] = int.from_bytes(all_data[i], byteorder='big')


def createTraining(spec_training, name):
    if (name == "cat"):
        for i in range(800):
            spec_training.append([])
            for j in range(784):
                spec_training[i].append(float(all_data[i * 784 + j]))
    if (name == "train"):
        for i in range(1000, 1800):
            spec_training.append([])
            for j in range(784):
                spec_training[i - 1000].append(float(all_data[i * 784 + j]))
    if (name == "rainbow"):
        for i in range(2000, 2800):
            spec_training.append([])
            for j in range(784):
                spec_training[i - 2000].append(float(all_data[i * 784 + j]))
    for i in range(800):
        spec_training[i].append(name)


def createTest(spec_test, name):
    if (name == "cat"):
        for i in range(800, 1000):
            spec_test.append([])
            for j in range(784):
                spec_test[i - 800].append(float(all_data[i * 784 + j]))
    if (name == "train"):
        for i in range(1800, 2000):
            spec_test.append([])
            for j in range(784):
                spec_test[i - 1800].append(float(all_data[i * 784 + j]))
    if (name == "rainbow"):
        for i in range(2800, 3000):
            spec_test.append([])
            for j in range(784):
                spec_test[i - 2800].append(float(all_data[i * 784 + j]))
    for i in range(800, 1000):
        spec_test[i - 800].append(name)


def normalizeValues(arr):
    for i in range(len(arr)):
        for j in range(784):
            arr[i][j] = float(arr[i][j])
            arr[i][j] /= 255.0


def main():
    # Creating NN
    nn = NeuralNetwork([784, 64, 3])
    # Creating training data
    createTraining(cats_training, "cat")
    createTraining(trains_training, "train")
    createTraining(rainbows_training, "rainbow")
    # Creating test data
    createTest(cats_testing, "cat")
    createTest(trains_testing, "train")
    createTest(rainbows_testing, "rainbow")
    # make one array from all the training and test data
    training_data = cats_training + trains_training + rainbows_training
    testing_data = cats_testing + trains_testing + rainbows_testing
    # Normalize the training data and testing data, (0-1)
    normalizeValues(trains_training)
    normalizeValues(testing_data)
    # arrange the data in specific format that will fit to the NN library i created.
    for j in range(2400):
        for i in range(785):
            training_data[j][i] = [training_data[j][i]]
    for j in range(600):
        for i in range(785):
            testing_data[j][i] = [testing_data[j][i]]

    '''
    training the model with the training data (30 times),
    shuffle the training data every iteration.
    specific target to every label.
    '''
    for i in range(35):
        random.shuffle(training_data)
        for j in range(2400):
            target = [[0], [0], [0]]
            if (training_data[j][784] == ['cat']):
                target[0] = [1]
                nn.train(training_data[j][:-1], target, 1)
            if (training_data[j][784] == ['train']):
                target[1] = [1]
                nn.train(training_data[j][:-1], target, 1)
            if (training_data[j][784] == ['rainbow']):
                target[2] = [1]
                nn.train(training_data[j][:-1], target, 1)
    '''
    Calculate the acc. should be more various of training data.
    currently : 800 cats, 800 rainbows, 800 trains.
    '''
    acc = 0
    trains = 0
    rainbows = 0
    cats = 0
    for i in range(len(testing_data)):
        label = testing_data[i][784]
        guess = nn.feed_forword(testing_data[i][:-1])
        if max(guess) == guess[0] and label == ['cat']:
            acc += 1
            cats += 1
        if max(guess) == guess[1] and label == ['train']:
            acc += 1
            trains += 1
        if max(guess) == guess[2] and label == ['rainbow']:
            acc += 1
            rainbows += 1
    print("Acc : " + str(acc))
    print("length of testing data : " + str(len(testing_data)))
    print("Cats : " + str(cats))
    print("Rainbows : " + str(rainbows))
    print("trains : " + str(trains))


main()
