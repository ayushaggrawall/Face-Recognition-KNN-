import numpy as np
from random import randint
import json
import pandas as pd
import pickle


def euc(v1,v2):
    s = 0
    for x,y in zip(v1,v2):
        s = s + (x-y)**2
    value = s**0.5
    return value

def get_embedding_dataframe(path):
    # df = pd.read_json(path, lines = True)
    # df['face_embeddings']=df['_source'].apply(lambda x: x['face_embeddings'])
    # dfaces = df["face_embeddings"]
    # del df
    # return pd.DataFrame(list(dfaces.values))
    with open('data.pkl', 'rb') as f:
        dfaces = pickle.load(f)
    return pd.DataFrame(list(dfaces.values))

def rp(x1,cluster_centers,dfaces,cluster_indices,id_data):
    rdist = []
    for v1 in cluster_centers:
        np1=np.array(v1)
        rdistval = euc(dfaces.iloc[x1],np1)
        rdist.append(rdistval)
    print("\nEuc distance between point ", x1," and centroids of each cluster")
    for ix,i in enumerate(rdist):
        print(ix,'\t-->\t',i)
    rdist=np.array(rdist)
    print('\n')
    idx=np.argmin(rdist)
    print('minimum distance ',np.min(rdist),' from cluster ',idx)
    for i, point in enumerate(dfaces.values):
        cluster_index = cluster_indices[i]
        # center = cluster_centers[cluster_index]
        if(cluster_index==idx):
            if(i==x1):
                print ('point:', i, '(id.',id_data[i] ,') must be in cluster', cluster_index)
                return True
    return False



def main(path):
    #----------------------------Trained Data----------------------------------------#
    with open("cluster_centers.json",'r') as fp1:
        cluster_centers = json.load(fp1)
    cluster_centers=np.array(cluster_centers)
    with open("cluster_indices.json",'r') as fp1:
        cluster_indices = json.load(fp1)
    with open("id_data.json",'r') as fp1:
        id_data = json.load(fp1)
    #--------------------------------------------------------------------------------#   
    #path = '/home/valuepitch/Desktop/non_news_rss.json'
    dfaces = get_embedding_dataframe(path)
    for _ in range(10):
        value = randint(0, len(dfaces))
        check=rp(value,cluster_centers,dfaces,cluster_indices,id_data)
        print(check)

if __name__ == "__main__":
    main(path)