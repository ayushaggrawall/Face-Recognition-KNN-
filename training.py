import json
import pandas as pd
import math
import numpy as np
import tensorflow as tf
import sys
import pickle

def input_fn():
    global dfaces
    return tf.train.limit_epochs(
      tf.convert_to_tensor(dfaces.values, dtype=tf.float32), num_epochs=1)

def get_embedding_dataframe(path):
    df = pd.read_json(path, lines = True)
    df['face_embeddings']=df['_source'].apply(lambda x: x['face_embeddings'])
    df['unique_id_index']=df['_source'].apply(lambda x: x['unique_id_index'])
    dfaces = df["face_embeddings"]
    dfaces2 = df["unique_id_index"]
    del df
    with open('data.pkl', 'wb') as f:
        pickle.dump(dfaces, f)
    return pd.DataFrame(list(dfaces.values)),pd.DataFrame(list(dfaces2.values))

def euc(v1,v2):
    s = 0
    for x,y in zip(v1,v2):
        s = s + (x-y)**2
    value = s**0.5
    return value

def train(path,dfaces,dfaces2):
    # dfaces=dfaces[:1000]
    num_clusters=math.ceil((len(dfaces)/2)**(1/2))
    kmeans = tf.estimator.experimental.KMeans(num_clusters=num_clusters, use_mini_batch=False)
    print('Forming ',num_clusters,' clusters')
    print("---------------SAVING MODEL----------------")
    # train
    num_iterations = 500
    previous_centers = []
    for i in range(num_iterations):
        kmeans.train(input_fn)
        cluster_centers = kmeans.cluster_centers()
        cluster_centers=np.array(cluster_centers)
        if np.array_equal(previous_centers,cluster_centers):
            print('Completed\n------------------------------------------------------------------------------------------------')
            print(i)
            break
        else:
            print('Iteration:',i)
        previous_centers = cluster_centers

    cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    cluster_centers =cluster_centers.tolist()
    with open('cluster_centers.json', 'w') as json_file:
        json.dump(cluster_centers, json_file)
    for ix,i in enumerate(cluster_indices):
        cluster_indices[ix]=int(cluster_indices[ix])
    with open('cluster_indices.json', 'w') as json_file:
        json.dump(cluster_indices, json_file)

    dfaces2 = dfaces2[0].values.tolist()
    with open('id_data.json', 'w') as json_file:
        json.dump(dfaces2, json_file)


def main(path):
    global dfaces
    global dfaces2
    dfaces,dfaces2 = get_embedding_dataframe(path)
    train(path,dfaces,dfaces2)

if __name__ == "__main__":
    main(path)
