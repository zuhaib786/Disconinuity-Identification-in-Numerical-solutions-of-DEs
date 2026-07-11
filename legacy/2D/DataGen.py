import numpy as np
import matplotlib.pyplot as plt
import math
class Legendre:
  def __init__(self):
    self.coeff = [[1], [0, 1], [-0.5, 0, 1.5], [0, -1.5, 0, 2.5], [0.375, 0, -3.75,0 , 4.375]]
  def calc(self, x, deg):
    data = self.coeff[deg]
    ans = 0
    mul = 1
    for i in data:
      ans += i * mul
      mul = mul * x
    return ans
class DataGenerator:
  def __init__(self, N_P = 4):
    self.N_P = 4
    self.L = Legendre()
    self.coeff_mat1  = self.genMat()
    self.coeff_mat2 = self.genMat()
    self.order = np.linspace(-1, 1, 101)
    self.mesh_x = np.linspace(-1, 1, 101)
    self.mesh_y = np.linspace(-1, 1, 101)
    self.mesh_x , self.mesh_y = np.meshgrid(self.mesh_x, self.mesh_y)
  def calc(self, mat):
    ans = np.zeros((101, 101))
    data = []
    if mat == 1:
      data = self.coeff_mat1
    else:
      data = self.coeff_mat2 

    for i in range(self.N_P + 1):
      for j in range(self.N_P  +1 - i):
        A = self.L.calc(self.mesh_x, i)
        B = self.L.calc(self.mesh_y, j)
        ans += data[i][j] *(A*B)
    return ans
  def lineCut(self):
    theta = np.random.uniform(0, 2*math.pi)
    x_0, y_0 = np.random.choice(self.order , 1), np.random.choice(self.order, 1)
    x_0 = list(x_0)[0]
    y_0 = list(y_0)[0]
    return self.lineToBinaryMat(x_0, y_0, theta) 
  def LineCalc(self, x_0, y_0, theta, x, y):
    return np.cos(theta) * (x - x_0) + np.sin(theta) * (y - y_0)
  def circCut(self):
    r = np.random.uniform(0.5, 3)
    x_0, y_0 = np.random.choice(self.order , 1), np.random.choice(self.order, 1)
    x_0 = list(x_0)[0]
    y_0 = list(y_0)[0]
    x_0, y_0 = 0,0
    return self.cricToBinaryMat(x_0, y_0, r)
  def lineToBinaryMat(self, x_0, y_0, theta):
    A = self.LineCalc(x_0, y_0, theta, self.mesh_x, self.mesh_y)
    labels = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            val1 = self.LineCalc(x_0, y_0, theta, self.mesh_x[i, j], self.mesh_y[i, j])
            val2 = self.LineCalc(x_0, y_0, theta, self.mesh_x[i + 1, j], self.mesh_y[i + 1, j])
            val3 = self.LineCalc(x_0, y_0, theta, self.mesh_x[i, j + 1], self.mesh_y[i, j + 1])
            val4 = self.LineCalc(x_0, y_0, theta, self.mesh_x[i + 1, j + 1], self.mesh_y[i + 1, j + 1])
            ct1, ct2  = 0, 0
            for val in [val1, val2, val3, val4]:
                if val <= 0:
                    ct1 += 1
                else:
                    ct2 += 1
            if ct1 != 0 and ct2 != 0:
                labels[i, j] = 1
    return A>=0 , labels

  def CircCalc(self, x_0, y_0, r, x, y):
    return (x - x_0) * (x- x_0) + (y - y_0) * (y - y_0) - r
  def cricToBinaryMat(self, x_0, y_0, r):
    A = self.CircCalc(x_0, y_0, r, self.mesh_x, self.mesh_y)
    labels = np.zeros((100, 100))
    for i  in range(100):
        for j in range(100):
            val1 = self.CircCalc(x_0, y_0,  r, self.mesh_x[i, j], self.mesh_y[i, j])
            val2 = self.CircCalc(x_0, y_0, r,self.mesh_x[i  + 1, j], self.mesh_y[i + 1, j])
            val3  = self.CircCalc(x_0, y_0, r, self.mesh_x[i, j+ 1], self.mesh_y[i, j + 1])
            val4 = self.CircCalc(x_0, y_0, r, self.mesh_x[i + 1, j + 1], self.mesh_y[i + 1, j + 1])
            ct1, ct2  = 0, 0
            for val in [val1, val2, val3, val4]:
                if val <= 0:
                    ct1 += 1
                else:
                    ct2 += 1
            if ct1 != 0 and ct2 != 0:
                labels[i, j] = 1
    return A>=0 , labels
  def genMat(self):
    data = np.random.normal(0, 1, size=(101, 101))
    return data
  def DataPointLine(self):
    self.mat1 = self.genMat()
    self.mat2 = self.genMat()
    ans1  = self.calc(1)
    ans2 = self.calc(2)
    y, label = self.lineCut()
    ans = ans1 * y + ans2 *( 1- y)
    return ans, label
  def DataPointCirc(self):
    self.mat1 = self.genMat()
    self.mat2 = self.genMat()
    ans1 = self.calc(1)
    ans2 = self.calc(2)
    y, label = self.circCut()
    ans = ans1 * y + ans2 * ( 1- y)
    return ans, label


