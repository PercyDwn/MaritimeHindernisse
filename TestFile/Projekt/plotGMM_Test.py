from plotGMM import *

print('create birth_belief gmm ...')
birth_belief = []
cov = array(
  [[1000, 0.0, 0.0, 0.0], 
   [0.0, 800.0, 0.0, 0.0],
   [0.0, 0.0, 10.0, 0.0],
   [0.0, 0.0, 0.0, 10.0]])
for i in range(0,5):
  for j in range(0,5):
    x = 20 + i*140
    y = 20 + j*100
    mean = vstack([x, y, 10, 2])
    birth_belief.append(Gaussian(mean, cov))
print('OK created')

fig = plotGMM(gmm = birth_belief, pixel_w = 640, pixel_h = 480, detail = 1 , method = 'rowwise', figureTitle = 'Birth Belief GMM', savePath = '')
fig.show()
fig = plotGMM(gmm = birth_belief, pixel_w = 640, pixel_h = 480, detail = 1 , method = 'rowwise', figureTitle = 'Birth Belief GMM', savePath = '')
fig.show()