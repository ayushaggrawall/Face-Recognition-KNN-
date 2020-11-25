import numpy as np
import pandas as pd
import json
import pickle

def get_embedding_dataframe():
    # df = pd.read_json(path, lines = True)
    # df['face_embeddings']=df['_source'].apply(lambda x: x['face_embeddings'])
    # dfaces = df["face_embeddings"]
    # del df
    # return pd.DataFrame(list(dfaces.values))


    with open('data.pkl', 'rb') as f:
        dfaces = pickle.load(f)
    return pd.DataFrame(list(dfaces.values))

def query_check(query,cluster_centers):
    query_dist=[]
    for i in cluster_centers:
        np1=np.array(i)
        query_distval = euc(query,np1)
        query_dist.append(query_distval)
    query_dist=np.array(query_dist)
    idx=np.argmin(query_dist)
    return idx

def euc(v1,v2):
    s = 0
    for x,y in zip(v1,v2):
        s = s + (x-y)**2
    value = s**0.5
    return value

def close(check,query,cluster_indices,dfaces,id_data):
    cluster_points=[]
    cluster_points_id=[]
    dis=[]
    ref=[]
    for ix,i in enumerate(cluster_indices):
        if i==check:
            cluster_points.append(ix)
            cluster_points_id.append(id_data[ix])
    for i in cluster_points:
        cp=np.array(dfaces.iloc[i])
        disv=euc(cp,query)
        dis.append(disv)
    return dis,cluster_points,cluster_points_id
    # return dis,cluster_points


def main(query):
    dfaces = get_embedding_dataframe()
    # query=dfaces.iloc[1152]
    topresult=10

    #Trained data
    with open("cluster_centers.json",'r') as fp1:
        cluster_centers = json.load(fp1)
    cluster_centers=np.array(cluster_centers)
    with open("cluster_indices.json",'r') as fp1:
        cluster_indices = json.load(fp1)
    with open("id_data.json",'r') as fp1:
        id_data = json.load(fp1)
    

    query=np.array(query)
    check=query_check(query,cluster_centers)
    dis,cluster_points,id_dp=close(check,query,cluster_indices,dfaces,id_data)
    list1, list2, list3 = (list(t) for t in zip(*sorted(zip(dis, cluster_points,id_dp))))
    # print("Distance between query vector and datapoints in cluster ",check)
    for i in  range(topresult):
        if list1[i]==0.0:
            print('Query point is ',list1[i],'\t\t\t away from point ',list2[i],' \t(id.',list3[i],')\t in cluster ',check)
        else:
            print('Query point is ',list1[i],'\t away from point ',list2[i],' \t(id.',list3[i],')\t in cluster ',check)



if __name__ == "__main__":
    main(query)
    