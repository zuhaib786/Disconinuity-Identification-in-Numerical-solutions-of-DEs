import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')
def factorial(m):
    ans = 1
    for i in range(1, m + 1):
        ans *= i
    return ans
class Annihilator:
    def __init__(self, X, fvalues,  md = 2):
        self.m, self.n, _ = X.shape
        self.fvalues = fvalues
        m, n, _ = X.shape
        self.md = ((md+2)*(md +1))//2
        self.d = md
        self.arr = X.reshape((m*n, 2))
        # for i in range(m):
        #     for j in range(n):
        #         self.arr[m*i + j] = X[i, j]
        self.index = NearestNeighbors(n_neighbors = self.md, algorithm = 'ball_tree').fit(self.arr)

    def getTriangle(self, index):
        """
        index = (i, j)
        This is how triangles are included as indices
        There are 2*(m - 1)*(n - 1) triangles
        j gives the base row number
        if i = 0 (mod 2) then we take two elements from base and one from top
        otherwise one from base and two from top are included 
        """
        i, j = index
        x1, x2,x3= None, None, None
        if i%2 == 0:
            i = i//2
            x1 = self.arr[self.m*j + i ]
            x2 = self.arr[self.m*j + i + 1]
            x3 = self.arr[self.m*(j + 1) +i]
        elif i%2 == 1:
            i  = (i + 1)//2
            x1 = self.arr[self.m*(j + 1) + i]
            x2 = self.arr[self.m*(j + 1) + i -1]
            x3 = self.arr[self.m * j + i]
        return (x1 + x2 + x3)/3
    def getIndices(self, index):
        centr = self.getTriangle(index)
        _, indices = self.index.kneighbors(np.array([centr]).astype(np.float128 ))
        indices = indices[0]
        indices = indices.tolist()
        indices.sort(key = lambda x: self.fvalues[x])
        return indices
    def label(self, index):
        """
        Function used for labelling the data
        """
        indices  = self.getIndices(index)
        pts = [self.arr[index] for index in indices]
        r = 1
        for i in range(1, self.md):
            if self.fvalues[indices[i]] - self.fvalues[indices[i - 1]] > self.fvalues[indices[r]] - self.fvalues[indices[r - 1]]:
                r =i
        
        constants = self.getConstants(indices)
        constants = constants[0]
        normFactor = 0
        sm = 0
        for i, index in enumerate(indices):
            if i <r:
                normFactor += constants[i]
            sm += constants[i] * self.fvalues[index]
        return sm/normFactor
    

    def getConstants(self, indices):
        """
        Method to solve the system of linear equations 
        We use np.linalg.solve method of numpy to solve the system efficiently
        """
        mat = self.createMatrix(indices)
        # gaussian = np.random.randn(*mat.shape)*0.000001
        # mat = mat + gaussian
        # print(mat)
        load = self.getLoadMatrix()
        return np.linalg.lstsq(mat, load,rcond=None)
    def getLoadMatrix(self):
        """
        Get the load matrix 
        load[m*i + j] = 0 if i + j < d
        else
        load[m*i + j] = factorial(i) * factorial(j)
        """
        arr = np.zeros((self.md, 1))
        itr = 0
        for i in range(self.d+1):
            for j in range(self.d+1):
                if i + j > self.d:
                    continue
                if i + j == self.d:
                    arr[itr] = factorial(i)*factorial(j)
                itr += 1
        return arr
    def createMatrix(self, indices):
        """
        Matrix of the system of equations formed.
        This is the main method of the detection process
        """
        arr = []
        for i in range(self.d+1):
            for j in range(self.d+1):
                if i+j > self.d:
                    continue
                temp =  []

                for index in indices:
                    pt = self.arr[index]
                    temp.append(pow(pt[0], i) * pow(pt[1], j))
                arr.append(temp)
        return np.array(arr)

