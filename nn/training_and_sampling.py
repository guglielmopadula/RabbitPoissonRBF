#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""
import meshio
import numpy as np
from sklearn.decomposition import KernelPCA
from ezyrb import Database,RBF
from sklearn.neighbors import KernelDensity
from tqdm import trange
data=np.load("data/data.npy")
data_train=data[:400].reshape(400,-1)
data_test=data[400:].reshape(200,-1)
transformer = KernelPCA(n_components=3, kernel='rbf',fit_inverse_transform=True)
transformer.fit(data_train)
Xred=transformer.transform(data_train)
db1=Database(data_train,Xred)
rbf_fwd=RBF()
rbf_fwd.fit(Xred,data_train)
print(np.linalg.norm(rbf_fwd.predict(Xred)-data_train)/np.linalg.norm(data_train))
print(np.linalg.norm(rbf_fwd.predict(transformer.transform(data_test))-data_test)/np.linalg.norm(data_test))
Xred=transformer.transform(data.reshape(600,-1))
kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(Xred)

NUMBER_SAMPLES=600


tets=np.load("data/tetras.npy")
def list_faces(t):
  t.sort(axis=1)
  n_t, m_t= t.shape 
  f = np.empty((4*n_t, 3) , dtype=int)
  i = 0
  for j in range(4):
    f[i:i+n_t,0:j] = t[:,0:j]
    f[i:i+n_t,j:3] = t[:,j+1:4]
    i=i+n_t
  return f
def area(vertices, triangles):
    v1 = vertices[triangles[:,0]]
    v2 = vertices[triangles[:,1]]
    v3 = vertices[triangles[:,2]]
    a = np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1) / 2
    return np.sum(a)


def extract_unique_triangles(t):
  _, indxs, count  = np.unique(t, axis=0, return_index=True, return_counts=True)
  return t[indxs[count==1]]

def extract_surface(t):
  f=list_faces(t)
  f=extract_unique_triangles(f)
  return f


temp=np.zeros(data.shape)
triangles=extract_surface(tets)
moment_tensor_sampled=np.zeros((NUMBER_SAMPLES,3,3))
area_sampled=np.zeros(NUMBER_SAMPLES)
name="RBF"

moment_tensor_data=np.zeros((NUMBER_SAMPLES,3,3))
area_data=np.zeros(NUMBER_SAMPLES)
error=0
for j in range(3):
    for k in range(3):
        moment_tensor_data[:,j,k]=np.mean(data.reshape(NUMBER_SAMPLES,-1,3)[:,:,j]*data.reshape(NUMBER_SAMPLES,-1,3)[:,:,k],axis=1)

for i in trange(NUMBER_SAMPLES):
    area_data[i]=area(data.reshape(NUMBER_SAMPLES,-1,3)[i],triangles)


for i in trange(NUMBER_SAMPLES):
    tmp=transformer.inverse_transform(kde.sample())
    tmp=tmp.reshape(-1,3)
    tmp=tmp-np.mean(tmp,axis=0)
    error=error+np.min(np.linalg.norm(tmp.reshape(-1)-data.reshape(NUMBER_SAMPLES,-1),axis=1))/np.linalg.norm(data.reshape(NUMBER_SAMPLES,-1))/NUMBER_SAMPLES
    area_sampled[i]=area(tmp,triangles)
    temp[i]=tmp.reshape(-1)
    meshio.write_points_cells("./nn/inference_objects/RBF_{}.ply".format(i), tmp,[])

print("Variance of RBF is", np.sum(np.var(temp.reshape(NUMBER_SAMPLES,-1),axis=0)))    
for j in range(3):
    for k in range(3):
        moment_tensor_sampled[:,j,k]=np.mean(temp.reshape(NUMBER_SAMPLES,-1,3)[:,:,j]*temp.reshape(NUMBER_SAMPLES,-1,3)[:,:,k],axis=1)
variance=np.sum(np.var(temp,axis=0))
variance_data=np.sum(np.var(data.reshape(NUMBER_SAMPLES,-1),axis=0))
np.save("nn/geometrical_measures/moment_tensor_data.npy",moment_tensor_data)
np.save("nn/geometrical_measures/moment_tensor_"+name+".npy",moment_tensor_sampled)
print("Saved moments")
np.save("nn/geometrical_measures/area_data.npy",area_data)
np.save("nn/geometrical_measures/area_"+name+".npy",area_sampled)
print("Saved Areas")
np.save("nn/geometrical_measures/variance_"+name+".npy",variance)
np.save("nn/geometrical_measures/variance_data.npy",variance_data)
print("Saved variances")
np.save("nn/geometrical_measures/rel_error_"+name+".npy",error)
print("Saved error")
np.save("nn/inference_objects/"+name+".npy",temp.reshape(NUMBER_SAMPLES,-1,3))
