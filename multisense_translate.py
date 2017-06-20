# coding: utf-8

from collections import defaultdict
import logging
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from gensim.models import KeyedVectors

logging.basicConfig(format="%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s", level=logging.DEBUG)

class MultiSenseLinearTranslator():
    def __init__(self,
                 source_mse_filen='/mnt/permanent/Language/Hungarian/Embed/multiprot/adagram/mnsz/'
                 'adagram-mnsz-600d-a.05-5p-m100_sense.mse',
                 target_embed='/mnt/permanent/Language/English/Embed/glove.840B.300d.gensim',
                 seed_filen='/mnt/store/makrai/data/language/hungarian/dict/wikt2dict-en-hu.by-freq'):
        logging.warning(
            'The order of the source and the target language in the arguments '
            'for embedding and in the words in the seed file is opposite')
        self.source_mse_filen = source_mse_filen
        self.source_firsts = self.get_first_vectors(source_mse_filen)
        self.target_embed = self.get_first_vectors(target_embed)
        self.seed_filen = seed_filen
        self.train_size = 5000
        self.test_size = 1000
        self.train_sr = np.zeros((self.train_size, self.source_firsts.syn0.shape[1]))
        self.train_tg = np.zeros((self.train_size, self.target_embed.syn0.shape[1]))
        self.restrict_vocab = 10000

    def get_first_vectors(self, filen):
        root, ext = os.path.splitext(filen)
        if ext == 'gensim':
            return KeyedVectors.load(filen)
        else:
            gens_fn = '{}.gensim'.format(root)
            if os.path.isfile(gens_fn):
                return KeyedVectors.load(gens_fn)
            else:
                embed = KeyedVectors.load_word2vec_format(filen)
                embed.save(gens_fn)
                return embed

    def main(self):
        with open(self.seed_filen) as self.seed_f:
            train_size = 0
            while train_size < self.train_size:
                tg, sr = self.seed_f.readline().strip().split()
                if sr in self.source_firsts and tg in self.target_embed:
                    self.train_sr[train_size] = self.source_firsts[sr]
                    self.train_tg[train_size] = self.target_embed[tg]
                    train_size += 1

            logging.info('trained on {} words'.format(len(self.train_sr)))
            self.regression = LinearRegression(n_jobs=-2)
            self.regression.fit(self.train_sr, self.train_tg)
            self.test()

    def translate_one_vector(self, tg_vecs):
        ambig_neighbors = set()
        for tg_vec in tg_vecs:
            sense_neighbors, _ = zip(*self.target_embed.similar_by_vector(
                tg_vec, restrict_vocab=self.restrict_vocab))
            ambig_neighbors |= set(sense_neighbors)
        return ambig_neighbors


    def test(self):
        test_dict = defaultdict(set)
        for line in self.seed_f.readlines():
            tg, sr = line.split()
            test_dict[sr].add(tg)
        with open(self.source_mse_filen) as source_mse:
            logging.info(
                'skipping header: {}'.format(source_mse.readline().strip()))
            test_size = 0
            sr_word = ''
            sr_vecs = []
            good_trans = set()
            while True:#test_size < self.test_size:
                line = source_mse.readline()
                if not line:
                    logging.warning(
                        'tested only on {} items'.format(test_size))
                    break
                new_sr_word, vect_str = line.strip().split(maxsplit=1)
                if new_sr_word != sr_word:
                    if sr_word in test_dict:
                        test_size += 1
                        sr_vecs = np.concatenate(sr_vecs)
                        tg_vecs = sr_vecs.dot(self.regression.coef_.T)
                        ambig_neighbors = self.translate_one_vector(tg_vecs)
                        good_trans = list(
                            test_dict[sr_word].intersection(ambig_neighbors))
                        if len(good_trans) > 1:
                            good_vecs = np.concatenate([
                                self.target_embed[w].reshape((1,-1))
                                for w in good_trans])
                            if len(good_vecs) > 2:
                                #logging.warning( 'Handling of more than two good translations is not tested')
                                sim_mx = good_vecs.dot(good_vecs.T)
                                ij = np.unravel_index(
                                    np.argmin(sim_mx), sim_mx.shape)
                                tgw1, tgw2 = [good_trans[i] for i in ij]
                                logging.debug((sr_word, good_trans, tgw1,
                                               tgw2))
                    sr_word = new_sr_word
                    sr_vecs = []
                sr_vecs.append(np.fromstring(vect_str, sep=' ').reshape((1,-1)))

    def test_baseline(self):
        test_size = 0
        score_at10 = 0
        while test_size < self.test_size:
            line = self.seed_f.readline().strip()
            try:
                tg, sr = line.split()
            except:
                logging.err(line)
            if sr in self.source_firsts:
                #if not test_size % 100:
                #    logging.info('testing... ({} tested)'.format(test_size))
                tg_vec = self.regression.coef_.dot(self.source_firsts[sr])
                tg_10, _ = zip(*self.target_embed.similar_by_vector(tg_vec,
                                                                       restrict_vocab=self.restrict_vocab))
                test_size += 1
                if tg in tg_10:
                    score_at10 += 1
                    if not score_at10 % 100:
                        logging.info(
                            'prec@10 {:%}, test size {} {} {}'.format(
                                float(score_at10)/test_size, test_size, sr,
                                tg_10))
        prec_at10 = float(score_at10)/test_size
        logging.info('final prec@10 is {:%} on {} items'.format(prec_at10, test_size))
        return prec_at10

if __name__ == '__main__':
    print(MultiSenseLinearTranslator(#target_embed=sys.argv[1]
                                    ).main())
