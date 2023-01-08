from CoarseDataGen import *
from tqdm.notebook import tqdm
import random
from CoarseModel import createCoarseModel
def getData(M):
  d = DataGenerator()
  x = []
  y = []
  for i in tqdm(range(M)):
    a = random.randint(1, 5)
    if a%2 == 0:
      ans, label, coarse_labels = d.DataPointLine()
    else:
      ans, label, coarse_labels = d.DataPointCirc()
    coarse_labels = coarse_labels.reshape(100)
    m,n = ans.shape
    ans = ans.reshape((m, n, 1))
    x.append(ans)
    y.append(coarse_labels)
  return np.asarray(x), np.asarray(y)
model = createCoarseModel()
for _ in tqdm(range(1000)):
  x,y = getData(1000)
  model.fit(x = x,
            y = y,
            validation_split =0.1, 
            epochs = 10
            )
model.save('Models/CoarseModel')
