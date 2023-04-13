import numpy as np

# Load the embeddings from embeddings.npy
embeddings = np.load('./embeddings.npy')
q_emb = np.load('./output-dir/query_embeddings.npy')
# Print the shape of the embeddings array
print("./emb.npy", embeddings.shape)
print("./q_emb.npy", q_emb.shape)
print(q_emb)
