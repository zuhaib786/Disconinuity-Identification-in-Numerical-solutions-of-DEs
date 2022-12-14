import numpy as np
import pandas as pd
def factorial(m):
    ans = 1
    for i in range(1, m + 1):
        ans *= i
    return ans
class Annihilator:
    def __init__(self, m = 5):
        self.m = m
        pass
    def predict(self, data):
        pass
    def getIndices(self, index, n):
        l = index 
        r = index + 1
        id = 1
        while(r - l < self.m + 1):
            if id == 0 and l -1 >=0:
                l -=1
            elif id == 1 and r < n :
                r += 1
            id ^=1
        return list(range(l, r))
    def label(self, index,data, fvalues):
        indices  = self.getIndices(index, len(data))
        constants = self.getConstants(indices, data)
        normFactor = 0
        sm = 0
        l_index =indices[0] 
        for i in indices:
            if i>= index:
                normFactor += constants[i - l_index]
            sm += constants[i - l_index] * fvalues[i]
        sm = sm/normFactor
        return sm

    def getConstants(self, indices, data):
        mat = self.createMatrix(indices, data)
        load = self.getLoadMatrix()
        return np.linalg.solve(mat, load)
    def getLoadMatrix(self):
        arr = np.zeros((self.m + 1, 1))
        arr[self.m] = factorial(self.m)
        return arr
    def createMatrix(self, indices, data):
        arr = []
        for i in range(self.m + 1):
            temp =  []
            for index in indices:
                temp.append(pow(data[index], i))
            arr.append(temp)
        return np.array(arr)

