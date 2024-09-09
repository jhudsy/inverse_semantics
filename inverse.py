import numpy as np
import networkx as nx

def inverse_hcat(graph,degrees):
  d=np.zeros(len(degrees))
  for deg in degrees:
    d[deg]=degrees[deg]
  adj=nx.convert_matrix.to_numpy_array(graph)
  adj=adj.T
  M=np.diag(d)
  return d+np.matmul(M,np.matmul(adj,d))

def inverse_card(graph,degrees):
  d=np.zeros(len(degrees))
  for deg in degrees:
    d[deg]=degrees[deg]

  adj=nx.convert_matrix.to_numpy_array(graph)
  adj=adj.T
  att=np.sum(adj,axis=1)
  att_m=np.diag(np.sum(adj,axis=1))
  div=np.zeros(len(d))
  np.divide(np.matmul(np.diag(d),np.matmul(adj,d)),att,where=att!=0,out=div)
  lhs=d+np.matmul(att_m,d)+div
  return lhs

def inverse_mbs(graph,degrees):
  adj=nx.convert_matrix.to_numpy_array(graph)
  adj=adj.T

  d=np.zeros(len(degrees))
  for deg in degrees:
    d[deg]=degrees[deg]
  M=np.diag(d)

  return d+np.matmul(M,np.max(np.matmul(adj,M),axis=1))
