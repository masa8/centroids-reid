import numpy as np
np.set_printoptions(threshold=np.inf)   # print entire array

def show(path):
    print("===========",path,"===========")
    # Load the embeddings from embeddings.npy
    embeddings = np.load('./embeddings.npy')
    print("./emb.npy", embeddings.shape)
    print(embeddings[0])
    paths = np.load('./paths.npy')
    print(paths)

    q_emb = np.load('./{}/query_embeddings.npy'.format(path), allow_pickle=True)
    q_pat = np.load('./{}/query_paths.npy'.format(path), allow_pickle=True)
    q_res = np.load('./{}/results.npy'.format(path), allow_pickle=True)
    # Print the shape of the embeddings array
    
    print("./q_emb.npy", q_emb.shape)
    print(q_emb)
    print("./q_pat.npy", q_pat.shape)
    print(q_pat)
    print("./q_res.npy", q_res.shape)
    print(q_res)


show('output-dir-cos')
show('output-dir-euc')
