"""
This is a more complex example on performing clustering on large scale dataset.
This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.
A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.
The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).
In this example, we download a large set of questions from Quora and then find similar questions in this set.
"""
from sentence_transformers import SentenceTransformer, util
import os
import csv
import time

from doc_embedding.utils import prepare_features
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from transformers import AutoTokenizer, AutoModel
from pyvi.ViTokenizer import tokenize
import torch


# link resource
# https://www.sbert.net/examples/applications/clustering/README.html
# Model for computing sentence embeddings. We use one trained for similar questions detection
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

dataset_path = "data/news_10k.csv"
dataset_path = "data/acb.csv"
max_corpus_size = 2000 # We limit our corpus to only the first 50k questions


# Get all unique sentences from the file
corpus_sentences = set()
with open(dataset_path, encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        corpus_sentences.add(row['DESCRIPTION'])
        if len(corpus_sentences) >= max_corpus_size:
            break

corpus_sentences = list(corpus_sentences)
# print(corpus_sentences)
print("Encode the corpus. This might take a while")
# =============== Sentence Tranformers ========================
# corpus_embeddings = model.encode(corpus_sentences,
#                                  batch_size=64,
#                                  show_progress_bar=True,
#                                  convert_to_tensor=True,
#                                  device="cpu")

# ================ SNCSE Embedding ============================
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("data/checkpoint-7000")
model = AutoModel.from_pretrained("data/checkpoint-7000")
model = model.to(device)


def get_embeddings(sentences, model, tokenizer, device):
    sentences_tokenizer = [tokenize(sentence) for sentence in sentences]
    batch = prepare_features(sentences_tokenizer, tokenizer=tokenizer, max_len=256)

    # Move to the correct device
    for k in batch:
        batch[k] = torch.tensor(batch[k]).to(device)

    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.last_hidden_state.cpu()
        embeddings = pooler_output[batch['input_ids'] == tokenizer.mask_token_id]
    return embeddings


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


start_time = time.time()
embeddings_list = []
for x in batch(corpus_sentences, 100):
    # corpus_embeddings.append()
    # print(x)
    embeddings_list.extend(get_embeddings(x, model, tokenizer, device).tolist())

corpus_embeddings = torch.Tensor(embeddings_list)

print("Embedding done after {:.2f} sec".format(time.time() - start_time))
# corpus_embeddings = get_embeddings(corpus_sentences, model, tokenizer, device)
# print(corpus_embeddings)

# =============== Start cluster input Tensor =========================
print("Start clustering")
start_time = time.time()

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(corpus_embeddings, min_community_size=5, threshold=0.7)
print(clusters)
total = 0
for cluster in clusters:
    total += len(cluster)
print('Total sentence in cluster', total)


print("Clustering done after {:.2f} sec".format(time.time() - start_time))

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    for sentence_id in cluster:
        print(corpus_sentences[sentence_id])
    # for sentence_id in cluster[0:2]:
    #     print("\t", corpus_sentences[sentence_id])
    # print("\t", "...")
    # for sentence_id in cluster[-2:]:
    #     print("\t", corpus_sentences[sentence_id])
