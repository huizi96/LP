import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import classification_report
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score,f1_score,roc_curve,roc_auc_score
def load_data():
    """Load dataset"""
    print('Loading dataset...')
    edge11 = pd.DataFrame(pd.read_csv('../data_gcn/train14.csv'))
    edge2 = pd.DataFrame(pd.read_csv('../data_gcn/test5.csv'))
    node_df_long=pd.DataFrame(pd.read_csv('../data_gcn/df_long_vector.csv'))
    node_df_short=pd.DataFrame(pd.read_csv('../data_gcn/df_short_vector.csv'))
    link=pd.DataFrame(pd.read_csv("../data_gcn/weaklink.csv"))
    atten=pd.DataFrame(pd.read_csv("../data_gcn/usersocialattention.csv"))
    print("ok")
    edge11=edge11.sample(frac=1).reset_index(drop=True)
    edge2=edge2.sample(frac=1).reset_index(drop=True)
    node_df=pd.concat([node_df_long,node_df_short.iloc[:,1:]],axis=1)
    edge1=edge11
    edge_all=pd.concat([edge1,edge2],ignore_index=True)
            
    e1=pd.merge(edge_all,link,on=["user1","user2"],how="left")
    atten.rename(columns={"id":"user2"},inplace=True)
    e1=pd.merge(e1,atten,on="user2",how="left")
    e1=e1.fillna(0)
    e1=e1[["quantity","quality"]] 
    print("2")

    node_shape=node_df.shape[0]
    idx_features=np.array(node_df)
    edges_label=np.array(edge_all)
    labels = np.array(edges_label[:, -1],dtype=np.float32)
    # build featuresi
    word=np.array(node_df.iloc[:,1:]).tolist()
    feature=np.array(node_df.iloc[:,0:1])
    feature=feature.flatten()
    key=feature.tolist()
    value=[i for i in range(0,len(key))]
    word_to_ix=dict(zip(key,value))
    embeds=nn.Embedding(len(key),100)
    features=[]
    for row in node_df.values:
            account_application_number_idx=torch.LongTensor([word_to_ix[row[0]]])
            account_application_number_idx=Variable(account_application_number_idx)
            account_application_number_embed=embeds(account_application_number_idx).flatten()
            features.append(np.around(account_application_number_embed.flatten().detach().numpy(),decimals=3))
    features = sp.csr_matrix(features, dtype=np.float32)

    # build graph
    idx = np.array(idx_features[:, 0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edge_label1=edge11[(edge11["label"]==1)]
    edges_unordered = np.array(edge_label1.iloc[:,0:2])
    edges = np.array(list(map(idx_map.get,edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    edges_idx=np.array(edge_all.iloc[:,0:2])
    edges_idx=np.array(list(map(idx_map.get,edges_idx.flatten())),dtype=np.int32).reshape(edges_idx.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(node_shape, node_shape),
                        dtype=np.float32)
						
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))
    #features = normalize(features)


    idx_train = range(edge1.shape[0])
    idx_val=range(edge1.shape[0],edge1.shape[0]+int(0.5*edge2.shape[0]))
    idx_test=range(edge1.shape[0]+int(0.5*edge2.shape[0]),edge1.shape[0]+edge2.shape[0])
    word=torch.FloatTensor(np.array(word))
    e1=torch.FloatTensor(np.array(e1))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.FloatTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, word, e1,labels, idx_train, idx_val, idx_test, edges_idx

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    p=np.zeros((len(output)))
    for i in range(0,len(output)):
        if output[i]>0.5:
            p[i]=1
        else:
            p[i]=0
    acc=accuracy_score(labels,p)
    return acc
            

def metric(preds, labels):
    get_ks = lambda y_pred,y_true: ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    print("ks= {:.4f}".format(get_ks(preds,labels)))
    preds[preds>=0.5]=1
    preds[preds<0.5]=0
    target_names = ['class 0', 'class 1']
    acc=accuracy_score(labels,preds)
    print("accuracy= {:.4f}".format(acc.item()))
    print(classification_report(labels, preds, target_names=target_names))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

