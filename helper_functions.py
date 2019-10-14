# Functions to count ocurrences for a category of a feature (NX)

# Count the number of instances inside the database whose feature X = category
# Ex: getNX('AIMC', 3, fdata)
def getNX(feature, category, data):
    count = 0
    for index in data.index:
        if data.loc[index][feature] == category:
            count = count + 1
    return count

# Count the number of instances inside the database whose feature X_1 = category_1 and feature X_2 = category_2
# Ex: getNCX('AIMC', 3, 'Obesidad', 1, fdata)
def getNCX(feature_1, category_1, feature_2, category_2, data):
    count = 0
    for index in data.index:
        if data.loc[index][feature_1] == category_1 and data.loc[index][feature_2] == category_2:
            count = count + 1
    return count

# Get a conditional probability P(F_1 = C_1 | F_2 = C_2)
def getCondProb(feature_1, category_1, feature_2, category_2, data):
    ncx = getNCX(feature_1, category_1, feature_2, category_2, data)
    nx = getNX(feature_2, category_2, data)
    if nx != 0:
        p = ncx / float(nx)
    else:
        p = 0
    return p

def getEpsilon(feature, category, classFeature, classCategory, data):
    n = len(data)
    nx = getNX(feature, category, data)
    nc = getNX(classFeature, classCategory, data)
    ncx = getNCX(feature, category, classFeature, classCategory, data)
    if n != 0 and nx != 0:
        pc = nc / float(n)
        pcx = ncx / float(nx)
        epsilon = nx * (pcx - pc) / math.sqrt(nx * pc * (1 - pc))
    else:
        epsilon = 0
    #print 'Epsilon :' + str(epsilon)
    return {'epsilon': epsilon, 'nx': nx, 'ncx': ncx, 'nc': nc}

# Functions to get the history of a certain individual
# The given feature must be a history-based parameter inside the database (e.g. salud, estres, condi, etc.)
def getRawHistory(index, feature, data):
    history = {}
    #history['index'] = index
    sufix_list = ['_act', '1', '5', '10', '20', '30']
    for sufix in sufix_list:
        history[feature + sufix] = data.loc[index][feature + sufix]
    return history

def binarizeHistory(history, conditionA, conditionB):
    for feature in history:
        if conditionA(history[feature]):
            history[feature] = 'A'
        elif conditionB(history[feature]):
            history[feature] = 'B'
        else:
            history[feature] = 'N'
    return history

# Set of auxiliary high-order functions that will evaluate the conditions to binarize a history
def lessThan(num):
    return lambda n: n < num

def lessQThan(num):
    return lambda n: n <= num

def greaterThan(num):
    return lambda n: n > num

def greaterQThan(num):
    return lambda n: n >= num

def between(num1, num2):
    return lambda n: n <= num2 and n >= num1

# Examples of use
history = getRawHistory(32, 'condi', fdata)
print history
binarizeHistory(history, lessThan(3), greaterQThan(3))

# Create a dictionary of binary histories for every individual in the database
# The feature must be a history-based parameter7
# conditionA and conditionB are the conditions of the binarization
def createDictionaryOfBinaryHistories(feature, conditionA, conditionB, data):
    histories = {}
    for index in data.index:
        current_raw_history = getRawHistory(index, feature, data)
        current_bin_history = binarizeHistory(current_raw_history, conditionA, conditionB)
        current_history = BinaryHistory(index, feature, current_bin_history)
        histories[index] = current_history
    return histories

# Get a list of all the individuals that have a certain type of binary history
def getIndividualsWithStructure(structure, histories):
    list_of_indeces = []
    for index in histories:
        if histories[index].historyStructureEqualsTo(structure):
            list_of_indeces.append(1)
        else:
            list_of_indeces.append(0)
    return list_of_indeces


# Add a list of structures of binary histories as features in the database
def addBHListOfStructuresAsFeatures(list_of_structures, histories, data):
    new_data = data.copy()
    for structure in list_of_structures:
        new_data = addBHStructureAsFeature(structure, histories, new_data)
    return new_data

# Add a structure of a binary history as a feature in the database
def addBHStructureAsFeature(structure, histories, data):
    new_data = data.copy()
    list_of_individuals = getIndividualsWithStructure(structure, histories)
    new_data.insert(new_data.shape[1], structure, list_of_individuals, True)
    return new_data

# Test cases
dicthist = createDictionaryOfBinaryHistories('estres', between(3, 6), lessThan(3), fdata)
#for id in dicthist:
#    print dicthist[id].structure

#getIndividualsWithStructure('AAAAAA', dicthist)
#ndata = addBHStructureAsFeature('AAAAAA', dicthist, fdata)
ndata = addBHListOfStructuresAsFeatures(['AAAAAA', '******', 'BBBBBB', '****BB', '****AA'], dicthist, fdata)
