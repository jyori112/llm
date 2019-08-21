import sys
import logging
from collections import defaultdict

import click
import numpy as np
import cupy
from tqdm import tqdm

import embeddings

logger = logging.getLogger(__name__)

@cli.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

@cli.command()
@click.option('--batchsize', type=int, default=1000)
@click.option('--n-nearest-neighbors', '-k', type=int, default=10)
def inv_nn(batchsize, n_nearest_neighbors):
    logger.info("Loading")
    words, vecs = embeddings.load_emb(sys.stdin)

    vecs = cupy.array(vecs)

    logger.info("Normalizing")
    vecs = vecs / cupy.linalg.norm(vecs, axis=1)[:, None]

    logger.info("Calculating")
    inv_nns = defaultdict(list)
    for start in tqdm(range(0, len(words), batchsize)):
        end = min(start + batchsize, len(words))

        # (BATCHSIZE, DIM)
        batch_vecs = vecs[start:end]

        batch_words = words[start:end]

        # (BATCHSIZE, VOCABSIZE)
        batch_sim = batch_vecs.dot(vecs.T)

        # (BATCHSIZE, k)
        rank = cupy.argsort(batch_sim, axis=1)[:, -1:n_nearest_neighbors-2:-1]

        rank = cupy.asnumpy(rank)

        for word, word_indicies in zip(words, rank):
            for word_idx in word_indicies:
                inv_nns[words[word_idx]].append(word)

    for word in inv_nns:
        print("{}\t{}".format(word, ' '.join(inv_nns[word])))


@cli.command()
@click.option('--batchsize', type=int, default=1000)
@click.option('--n-nearest-neighbors', '-k', type=int, default=10)
def main(batchsize, n_nearest_neighbors):
    logger.info("Loading")
    words, vecs = embeddings.load_emb(sys.stdin)

    vecs = cupy.array(vecs)

    logger.info("Normalizing")
    vecs = vecs / cupy.linalg.norm(vecs, axis=1)[:, None]

    nn_sim = np.empty((len(words), ), dtype=np.float32)

    logger.info("Calculating")
    for start in tqdm(range(0, len(words), batchsize)):
        end = min(start + batchsize, len(words))

        # (BATCHSIZE, DIM)
        batch_vecs = vecs[start:end]

        # (BATCHSIZE, VOCABSIZE)
        batch_sim = batch_vecs.dot(vecs.T)
        #print(batch_sim)

        # (BATCHSIZE, k)
        sorted_sim = cupy.sort(batch_sim, axis=1)[:, ::-1]
        batch_nn_sim = sorted_sim[:, 1:n_nearest_neighbors+1]

        # (BATCHSIZE, )
        batch_nn_sim_mean = cupy.mean(batch_nn_sim, axis=1)
        #print(batch_nn_sim_mean)

        nn_sim[start:end] = cupy.asnumpy(batch_nn_sim_mean)

    for i, word in enumerate(words):
        print("{}\t{}".format(word, nn_sim[i]))

if __name__ == '__main__':
    main()
