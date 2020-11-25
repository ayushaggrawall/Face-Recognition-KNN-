# Face-Recognition-KNN-

Main file to run "main_p.py". Different parts have been commented out. Uncomment to execute them. Instructions and what they are mentioned in comments.
Training results in "cluster_centers.json", "cluster_incides.json" , "id_data.json"
Convert json to pickle (.PKL) file for faster training and execution

Training data must be in format [_source][face_embeddings] and [_source][unique_id_index]

Make necessary changes to training data as deemed. 

#----------Import this module if you want to train KMeans clustering----------
# import training
# training.main(path)

#----------Get Euc. dist. between cluster centroids and also get minimum dist. among two centroids of clusters----------
import euc_dist
euc_dist.main()

#----------To verify if cluster points are in centroid----------
# import verify
# verify.main(path)

#----------Pass a query face (128 bit) to find nearest cluster and check nearest datapoint----------
import query
#format is in main_p.py
 


