import numpy as np

import json


def euc(v1,v2):
    s = 0
    for x,y in zip(v1,v2):
        s = s + (x-y)**2
    value = s**0.5
    return value

def main():
    #----------------------------Trained Data-----------------------------------#
    with open("cluster_centers.json",'r') as fp1:
        cluster_centers = json.load(fp1)
    cluster_centers=np.array(cluster_centers)
    with open("cluster_indices.json",'r') as fp1:
        cluster_indices = json.load(fp1)
    with open("id_data.json",'r') as fp1:
        id_data = json.load(fp1)
    #---------------------------------------------------------------------------#   

    dist = []
    distmat = []
    for v1 in cluster_centers:
        dist = []
        for v2 in cluster_centers:
            np1=np.array(v1)
            np2=np.array(v2)
            distval = euc(np1,np2)
            dist.append(distval)
        distmat.append(dist) 

    print('\n\n# Euc. Distances matrix between each cluster')
    for ix,i in enumerate(distmat):
        print('\n',ix,'-->',i)

    # Min. Euc. Dist. between clusters
    a=np.array(distmat)
    print('\n\n')
    for ix,i in enumerate(a):
        b=np.array(i)
        minval = np.min(b[np.nonzero(b)])
        index = np.argmin(b[np.nonzero(b)])
        print(ix,'\t--->\t',minval,'\t-->\t',index)


if __name__ == "__main__":
    main()

