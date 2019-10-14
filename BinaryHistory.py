# Binary History Class definition
class BinaryHistory:
    def __init__(self, index, feature, history):
        self.index = index
        self.feature = feature
        self.history = history
        self.structure = self.getHistoryStructure(history)
        self.complete = self.isHistoryComplete(history)
        self.length = self.getHistoryLength(history)

    def isHistoryComplete(self, history):
        return True

    def getHistoryLength(self, history):
        return 0

    def getHistoryStructure(self, history):
        time_list = ['30', '20', '10', '5','1','_act']
        structure = ''
        for t in time_list:
            structure = structure + self.history[self.feature + t]
        return structure

    def historyStructureEqualsTo(self, structure):
        for i in range(len(structure)):
            if self.structure[i] != structure[i] and structure[i] != '*':
                return False
        return True

# Test cases
test_index = 1
fdata = pd.read_csv('data_histories.csv', index_col="dp_folio")
history = getRawHistory(test_index, 'condi', fdata)
bhistory = binarizeHistory(history, lessThan(3), between(3, 6))

h1 = BinaryHistory(test_index, 'condi', bhistory)
print h1.structure
h1.historyStructureEqualsTo('B*B*BA')
