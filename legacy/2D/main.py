from CoarseDataGen import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from polAnnihilation2D import Annihilator
from scipy import io
d = DataGenerator()
ans, labels, coarse_labels = d.DataPointCirc()
arr = np.linspace(-1, 1, 101)
X = np.zeros((101, 101, 2))
for i in range(101):
    for  j in range(101):
        X[i, j, 0] = arr[i]
        X[i, j, 1] = arr[j]
annihilators = [Annihilator(X, ans.ravel(),i) for i in range(2, 7)]
m, n = ans.shape
pred =np.zeros((100, 100))
for i in range(0, 2*(m - 1), 2):
    for j in range(n - 1):
        d1 = [an.label((i, j)) for an in annihilators]
        mn = min(d1)
        mx = max(d1)
        if mn < 0 and mx < 0:
            l1 = mx
        elif mn > 0 and mx >0:
            l1 = mn
        else:
            l1 = 0
        d2 = [an.label((i + 1, j)) for an in annihilators]
        mn = min(d2)
        mx = max(d2)
        if mn < 0 and mx < 0:
            l2 = mx
        elif mn > 0 and mx >0:
            l2 = mn
        else:
            l2 = 0
        if abs(l1) >0.28 or abs(l2) >0.28:
            pred[i//2, j] = 1

x = np.linspace(-1, 1, 101)
y = np.linspace(-1, 1, 101)
x, y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(d.mesh_x, d.mesh_y, ans,color = 'black', s =1)
# ax.view_init(60, 35)


# fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(d.mesh_x[:100, :100], d.mesh_y[:100, :100], labels,color = 'black', s = 1)
ax.view_init(60, 35)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x[:100, :100], y[:100, :100], pred,color = 'black', s = 1)
ax.view_init(15, 35)

plt.show()
#