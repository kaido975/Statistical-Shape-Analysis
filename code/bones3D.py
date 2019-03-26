#%%
#libraries
import numpy as np
import h5py 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

#%%
#%matplotlib auto
n = 3
f = h5py.File('../data/bone3D.mat','r') 
shapesTotal = f.get('shapesTotal')
shapesTotal = np.array(shapesTotal)
TriangleIndex = f.get('TriangleIndex')
TriangleIndex = np.array(TriangleIndex)
TriangleIndex = TriangleIndex.astype(int)
TriangleIndex = TriangleIndex - 1
mean = np.mean(shapesTotal, axis=1)
mean = np.reshape(mean,(30,1,3))
pointSetsCen = shapesTotal - mean

for i in range(30):
    norm = np.linalg.norm(pointSetsCen[i, :, :])
    pointSetsCen[i, :, :] = pointSetsCen[i, :, :]/norm

#%%
#mean shape computation     
mean_shape = np.copy(pointSetsCen[0, :, :])
thresh = 1e-5
error = 1
while thresh < error:
    for i in range(30):
        a1 = np.matmul(pointSetsCen[i, :, :].T  , mean_shape)  
        u, s, vh = np.linalg.svd(a1)
        
        d = np.linalg.det(np.matmul(vh, u.T))
        eye = np.identity(n)
        eye[n-1,n-1] = -1
        R1 = vh @ u
        if np.linalg.det(R1)==-1:
            R1 = vh @ eye @ u         
        pointSetsCen[i, :, :] =  (R1 @ pointSetsCen[i, :, :].T).T
    
    best_mean_shape = np.mean(pointSetsCen, axis=0)  
#    plt.plot(best_mean_shape[:,0], best_mean_shape[:,1])
    best_mean_shape = best_mean_shape/np.linalg.norm(best_mean_shape)
    error = np.linalg.norm(best_mean_shape - mean_shape)
    
    mean_shape = best_mean_shape 
    
triang = mtri.Triangulation(mean_shape[:, 2],mean_shape[:, 1], TriangleIndex.T)
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Mean Shape')
ax.plot_trisurf(triang, mean_shape[:, 0], lw=0.05, edgecolor="black", color="red",
                alpha=1)
#%%
#Variance    
pointSetsCenNew = pointSetsCen - mean_shape
pointSetsCenNew = np.reshape(pointSetsCenNew, (30, 252*3))
covariance = np.cov(pointSetsCenNew.T) 
W, V = np.linalg.eig(covariance)
W, V = np.real(W), np.real(V)

plt.figure(2)
plt.plot(W)
#Principal Modes of shape Variation
s1 = np.sqrt(W[0])
s2 = np.sqrt(W[1])
s3 = np.sqrt(W[2])

a1 = mean_shape + 2*s1*np.reshape(V[:, 0],(252,3))
a2 = mean_shape - 2*s1*np.reshape(V[:, 0],(252,3))

b1 = mean_shape + 2*s2*np.reshape(V[:, 1],(252,3))
b2 = mean_shape - 2*s2*np.reshape(V[:, 1],(252,3))

c1 = mean_shape + 2*s3*np.reshape(V[:, 2],(252,3))
c2 = mean_shape - 2*s3*np.reshape(V[:, 2],(252,3))

#%%
#1st mode of variation
triang = mtri.Triangulation(a1[:, 2],a1[:, 1], TriangleIndex.T)
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('+2 std')
ax.plot_trisurf(triang, a1[:, 0], lw=0.05, edgecolor="black", color="blue",
                alpha=1)


triang = mtri.Triangulation(a2[:, 2],a2[:, 1], TriangleIndex.T)
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('-2 std')
ax.plot_trisurf(triang, a2[:, 0], lw=0.05, edgecolor="black", color="orange",
                alpha=1)

#%%
#2nd mode of variation
triang = mtri.Triangulation(b1[:, 2],b1[:, 1], TriangleIndex.T)
fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('+2 std')
ax.plot_trisurf(triang, b1[:, 0], lw=0.05, edgecolor="black", color="green",
                alpha=1)

triang = mtri.Triangulation(b2[:, 2],b2[:, 1], TriangleIndex.T)
fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('-2 std')
ax.plot_trisurf(triang, b2[:, 0], lw=0.05, edgecolor="black", color="violet",
                alpha=1)

#%%
#3rd mode of variation
triang = mtri.Triangulation(c1[:, 2],c1[:, 1], TriangleIndex.T)
fig = plt.figure(7)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('+2 std')
ax.plot_trisurf(triang, c1[:, 0], lw=0.05, edgecolor="black", color="green",
                alpha=1)

triang = mtri.Triangulation(c2[:, 2],c2[:, 1], TriangleIndex.T)
fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('-2 std')
ax.plot_trisurf(triang, c2[:, 0], lw=0.05, edgecolor="black", color="violet",
                alpha=1)
#%%%