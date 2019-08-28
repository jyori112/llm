import sys
import logging

import click
import numpy as np
import cupy
from cupy import cuda
from tqdm import tqdm

logger = logging.getLogger(__name__)
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def _as_batch_mat(x):
    return x.reshape(len(x), x.shape[1], -1)

def _mat_ptrs(a):
    if len(a) == 1:
        return cupy.full((1,), a.data.ptr, dtype=np.uintp)
    else:
        stride = a.strides[0]
        ptr = a.data.ptr
        out = cupy.arange(ptr, ptr + stride * len(a), stride, dtype=np.uintp)
        return out


def _get_ld(a):
    strides = a.strides[-2:]
    trans = np.argmin(strides)
    return trans, int(max(a.shape[trans - 2], max(strides) // a.itemsize))


def inv_gpu(b):
    # We do a batched LU decomposition on the GPU to compute the inverse
    # Change the shape of the array to be size=1 minibatch if necessary
    # Also copy the matrix as the elments will be modified in-place
    a = _as_batch_mat(b).copy()
    n = a.shape[1]
    n_matrices = len(a)
    # Pivot array
    p = cupy.empty((n, n_matrices), dtype=np.int32)
    # Output array
    c = cupy.empty_like(a)
    # These arrays hold information on the execution success
    # or if the matrix was singular
    info = cupy.empty(n_matrices, dtype=np.int32)
    ap = _mat_ptrs(a)
    cp = _mat_ptrs(c)
    _, lda = _get_ld(a)
    _, ldc = _get_ld(c)
    handle = cuda.Device().cublas_handle
    cuda.cublas.sgetrfBatched(
        handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
    cuda.cublas.sgetriBatched(
        handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,
        info.data.ptr, n_matrices)
    return c

def load_emb(f):
    n_vocab, dim = f.readline().strip().split()
    n_vocab, dim = int(n_vocab), int(dim)
    matrix = np.empty((n_vocab, dim), dtype=cupy.float32)
    word2id = {}
    id2word = {}

    for i, line in enumerate(f):
        word, vec_str = line.split(' ', 1)
        word2id[word] = i
        id2word[i] = word
        matrix[i] = np.fromstring(vec_str, sep=' ')

    return word2id, id2word, matrix


class Mapper:
    def __init__(self, words, gen_emb, spec_emb, num_neighbors, ignore_exact_words=False):
        self._words = words
        self._word2id = {word: i for i, word in enumerate(words)}
        self._gen_emb = gen_emb
        self._gen_emb_norm = gen_emb / np.linalg.norm(gen_emb, axis=1)[:, None]
        self._spec_emb = spec_emb
        self._num_neighbors = num_neighbors
        self._ignore_exact_words = ignore_exact_words
        self.output_dim = self._spec_emb.shape[1]

    def to_gpu(self):
        self._gen_emb = cupy.array(self._gen_emb)
        self._gen_emb_norm = cupy.array(self._gen_emb_norm)
        self._spec_emb = cupy.array(self._spec_emb)

    def get_neighobors(self, batch_emb, batch_words):
        # Normalize batch embeddings
        batch_emb_norm = batch_emb / cupy.linalg.norm(batch_emb, axis=1)[:, None]

        # Compute cosine similarity
        cos_score = batch_emb_norm.dot(self._gen_emb_norm.T)

        # Ignore exact matching words
        if self._ignore_exact_words:
            # indexはbatchの各単語のword indexをもっている
            word_index = cupy.array([self._word2id[word] for word in batch_words if word in self._word2id], dtype=cupy.int32)
            batch_index = cupy.array([i for i, word in enumerate(batch_words) if word in self._word2id], dtype=cupy.int32)

            # Set the score of matching words to very small
            cos_score[batch_index, word_index] = -100

        return cupy.argsort(-cos_score, axis=1)[:, :self._num_neighbors]

    def induce_weights(self, batch_emb, nn_idx):
        nn_gen_emb = self._gen_emb[nn_idx]

        diff = batch_emb[:, None] - nn_gen_emb
        C = cupy.einsum('ijk,ilk->ijl', diff, diff)
        C_inv = inv_gpu(C)

        w = cupy.sum(C_inv, axis=1) / cupy.sum(C_inv, axis=(1, 2))[:, None]

        return w

    def __call__(self, batch_emb, batch_words):
        nn_idx = self.get_neighobors(batch_emb, batch_words)

        weights = self.induce_weights(batch_emb, nn_idx)

        nn_spec_emb = self._spec_emb[nn_idx]
        ret = cupy.einsum('ijk,ij->ik', nn_spec_emb, weights)

        return ret

@click.command()
@click.argument('gen_emb_path', type=click.Path(exists=True))
@click.argument('spec_emb_path', type=click.Path(exists=True))
@click.option('--num-neighbors', type=int)
@click.option('--batchsize', type=int, default=100)
@click.option('--ignore-exact-words/--no-ignore-exact-words', default=False)
def main(gen_emb_path, spec_emb_path, num_neighbors, ignore_exact_words=False, batchsize=1000):
    logger.info("Load embeddings....")
    with open(gen_emb_path) as f:
        gen_word2id, _, gen_emb = load_emb(f)

    with open(spec_emb_path) as f:
        spec_word2id, _, spec_emb = load_emb(f)

    # 語彙の共通部分の抽出する
    logger.info("Extracting intersection...")
    gen_words = set(gen_word2id.keys())
    spec_words = set(spec_word2id.keys())
    words = list(gen_words.intersection(spec_words))

    # 語彙の共通部分をindex化して、用いるword embeddingsを決定
    gen_emb = gen_emb[[gen_word2id[word] for word in words]]
    spec_emb = spec_emb[[spec_word2id[word] for word in words]]

    # GPUへ転送
    #logger.info("Sending embeddings to GPU...", file=sys.stderr)
    #gen_emb = cupy.array(gen_emb)
    #spec_emb = cupy.array(spec_emb)

    # Mapping Modelを作成
    logger.info("Creating mapping model...")
    mapper = Mapper(words, gen_emb, spec_emb, num_neighbors, ignore_exact_words)
    mapper.to_gpu()

    # 標準入力から分散表現を読んでLLM
    batch_emb = cupy.empty((batchsize, gen_emb.shape[1]), dtype=cupy.float32)
    batch_words = []

    logger.info("Mapping...")
    n_vocab, _ = map(int, sys.stdin.readline().split())

    print("{} {}".format(n_vocab, mapper.output_dim))

    for i, line in tqdm(enumerate(sys.stdin), total=n_vocab):
        # embeddingをロード
        word, vec_str = line.split(' ', 1)
        batch_words.append(word)
        batch_emb[i % batchsize] = cupy.array(np.fromstring(vec_str, sep=' '))

        # batchsizeごとにmapして出力
        if i % batchsize == batchsize - 1:
            batch_spec_emb = mapper(batch_emb, batch_words)

            for k, word in enumerate(batch_words):
                vec_str = ' '.join('{:.6f}'.format(v) for v in cupy.asnumpy(batch_spec_emb[k]))
                print('{} {}'.format(word, vec_str))

            batch_words = []

    # 余ってしまった分を写像
    batch_spec_emb = mapper(batch_emb, batch_words)

    for i, word in enumerate(batch_words):
        vec_str = ' '.join('{:.6f}'.format(v) for v in cupy.asnumpy((batch_spec_emb[i])))
        print('{} {}'.format(word, vec_str))

if __name__ == '__main__':
    main()
