#%%
#libraries
import h5py 
import numpy as np
import matplotlib.pyplot as plt
#%%
#%matplotlib auto
n = 2
f = h5py.File('../data/ellipses2D.mat','r') 
numOfPoints = f.get('numOfPoints') 
numOfPoints = np.array(numOfPoints) # For converting to numpy array
numOfPointSets = f.get('numOfPointSets')
numOfPointSets = np.array(numOfPointSets)
pointSets = f.get('pointSets')
pointSets = np.array(pointSets)
plt.figure()
for i in range(300):
    plt.scatter(pointSets[i, :, 0], pointSets[i, :, 1])
plt.title('Plot of the initial pointsets, as given in the dataset', fontdict = {'fontsize' : 20})    
#%%
#mean shape computation
mean = np.mean(pointSets, axis=1)
mean = np.reshape(mean,(300,1,2))
pointSetsCen = pointSets - mean

for i in range(300):
    norm = np.linalg.norm(pointSetsCen[i, :, :])
    pointSetsCen[i, :, :] = pointSetsCen[i, :, :]/norm
    
#rotated = np.zeros((300,32,2))
mean_shape = np.copy(pointSetsCen[0, :, :])
thresh = 1.e-7
error = 1
while thresh < error:
#for j in range(2):   
    for i in range(300):
        a1 = np.matmul(pointSetsCen[i, :, :].T  , mean_shape)  
        u, s, vh = np.linalg.svd(a1, full_matrices=True)
    #    d = np.linalg.det(np.matmul(vh, u.T))
        eye = np.identity(n)
        eye[n-1,n-1] = -1
        R1 = vh @ u
        if np.linalg.det(R1)==-1:
            R1 = vh @ eye @ u   
        pointSetsCen[i, :, :] =  (R1 @ pointSetsCen[i, :, :].T).T
    
    best_mean_shape = np.mean(pointSetsCen, axis=0)  
    best_mean_shape = best_mean_shape/np.linalg.norm(best_mean_shape)
    error = np.linalg.norm(best_mean_shape - mean_shape)
    mean_shape = best_mean_shape    

plt.figure()
for i in range(300):
    plt.scatter(pointSetsCen[i, :, 0], pointSetsCen[i, :, 1])

plt.plot(best_mean_shape[:,0], best_mean_shape[:,1],c='black',linewidth=3, markersize=12)
plt.title('Plot of computed shape mean, together with all the aligned pointsets.', fontdict = {'fontsize' : 20})    
#%%
##Variance
pointSetsCenNew = pointSetsCen - mean_shape
pointSetsCenNew = np.reshape(pointSetsCenNew, (300, 64))
covariance = np.cov(pointSetsCenNew.T) 
W, V = np.linalg.eig(covariance)
W, V = np.real(W), np.real(V)


plt.figure()
plt.plot(W)
plt.title('Plot of the variances',fontdict = {'fontsize' : 20})
#Principal Modes of shape Variation
s1 = np.sqrt(W[0])
s2 = np.sqrt(W[1])

a1 = mean_shape + 2*s1*np.reshape(V[:, 0],(32,2))
a2 = mean_shape - 2*s1*np.reshape(V[:, 0],(32,2))

b1 = mean_shape + 2*s2*np.reshape(V[:, 1],(32,2))
b2 = mean_shape - 2*s2*np.reshape(V[:, 1],(32,2))

#%%
#1st mode of variation
plt.figure()
plt.plot(a1[:,0],a1[:,1], '-o')
plt.plot(a2[:,0],a2[:,1], '-o')
plt.plot(mean_shape[:,0], mean_shape[:,1], '-o')
plt.title('1st Mode of variation.',fontdict = {'fontsize' : 20})

plt.figure()
for i in range(300):
    plt.scatter(pointSetsCen[i, :, 0], pointSetsCen[i, :, 1])
plt.plot(a1[:,0],a1[:,1], '-o',c='black',linewidth=3)
plt.plot(a2[:,0],a2[:,1], '-o',c='black',linewidth=3)
plt.plot(mean_shape[:,0], mean_shape[:,1], '-o',c='black',linewidth=3)
plt.title('1st Mode of variation, together with all the aligned pointsets.',fontdict = {'fontsize' : 20})
#%%
#2nd mode of variation
plt.figure()
plt.plot(b1[:,0],b1[:,1],'-o')
plt.plot(b2[:,0],b2[:,1], '-o')
plt.plot(mean_shape[:,0], mean_shape[:,1], '-o')
plt.title('2nd Mode of variation.',fontdict = {'fontsize' : 20})


plt.figure()
for i in range(300):
    plt.scatter(pointSetsCen[i, :, 0], pointSetsCen[i, :, 1])
plt.plot(b1[:,0],b1[:,1], '-o',c='black',linewidth=3)
plt.plot(b2[:,0],b2[:,1], '-o',c='black',linewidth=3)
plt.plot(mean_shape[:,0], mean_shape[:,1], '-o',c='black',linewidth=3)
plt.title('2nd Mode of variation, together with all the aligned pointsets.',fontdict = {'fontsize' : 20})
