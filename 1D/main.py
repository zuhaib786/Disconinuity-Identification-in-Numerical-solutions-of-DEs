from polynomialAnnhilation import Annihilator
import numpy as np
import matplotlib.pyplot as plt
from OneDimData import DiscontDataGenerator
import tensorflow as tf
def f(x):
    if x<-0.9:  
        return np.sin(x)
    if x<-0.8:
        return 2*np.sin(x)
    if x<-0.7:
        return 3*np.sin(x)
    if x<-0.6:
        return 4*np.sin(x)
    if x<-0.5:
        return 5*np.sin(x)
    if x<-0.4:
        return 7*np.sin(x)
    if x<0.4:
        return 8*np.sin(x)
    return 6*np.sin(x)
an = Annihilator(5)
data = np.linspace(-1, 1, 201)
dg = DiscontDataGenerator(201)
fvalues, label = dg.generate()
fvalues = np.array(fvalues)
fun = np.frompyfunc(f, 1, 1)
fvalues = fun(data)
fvalues = fvalues.astype(np.float32)
flabel = []
for i in range(1, len(data)):
    flabel.append(an.label(i, data, fvalues))
flabel = np.array(flabel)
flabel = abs(flabel) > 1

plt.plot(data, fvalues)
kmodel = tf.keras.models.load_model('ModelDense1D')
mean = np.mean(fvalues)
std_dev = np.std(fvalues)
fvalues = (fvalues - mean)/(std_dev)
fvalues = fvalues.reshape((1, len(data), 1))

pred = kmodel.predict(fvalues)
fvalues = fvalues.reshape(201, )
pred = pred.reshape(200,)
x = np.linspace(-1, 1, 201)


x = x[:-1]
plt.plot(x, pred)
plt.plot(data[1:], flabel)
plt.legend(["Fun", "CNN", "Pol"])
plt.show()

