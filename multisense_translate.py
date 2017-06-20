# coding: utf-8

from collections import defaultdict
from functools import reduce
import logging
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from gensim.models import KeyedVectors

logging.basicConfig(
    format="%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s",
    level=logging.DEBUG)

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
            self.train()
            self.test()

    def train(self):
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

    def test(self):
        self.test_dict = self.read_test_dict()
        with open(self.source_mse_filen) as source_mse:
            logging.info(
                'skipping header: {}'.format(source_mse.readline().strip()))
            self.test_size = 0
            self.score_at10 = 0
            self.good_ambig = 0
            sr_word = ''
            sr_vecs = []
            while True:#self.test_size < self.test_size:
                line = source_mse.readline()
                if not line:
                    logging.warning(
                        'tested only on {} items'.format(self.test_size))
                    break
                new_sr_word, vect_str = line.strip().split(maxsplit=1)
                if new_sr_word != sr_word:
                    if sr_word in self.test_dict:
                        self.eval_word(sr_word, sr_vecs)
                    sr_word = new_sr_word
                    sr_vecs = []
                sr_vecs.append(np.fromstring(vect_str, sep=' ').reshape((1,-1)))

    def read_test_dict(self):
        with open(self.seed_filen) as seed_f:
            self.test_dict = defaultdict(set)
            for line in seed_f.readlines():
                tg, sr = line.split()
                self.test_dict[sr].add(tg)
            return self.test_dict 

    def eval_word(self, sr_word, sr_vecs): 
        self.log_prec() 
        self.test_size += 1
        tg_vecs = np.concatenate(sr_vecs).dot(self.regression.coef_.T)
        word_neighbors = [self.neighbors_by_vector(v) for v in tg_vecs]
        good_trans = [ns.intersection(self.test_dict[sr_word]) 
                      for ns in word_neighbors]
        all_good_trans = reduce(set.union, good_trans, set())
        if all_good_trans:
            self.score_at10 += 1 
            if len(all_good_trans) > 1:
                self.good_ambig += 1
                good_ambig_trans = [s for s in good_trans if s]
                if len(good_ambig_trans) > 1:
                    logging.debug((sr_word, good_ambig_trans))
    
    def neighbors_by_vector(self, vect):
        sense_neighbors, _ = zip(*self.target_embed.similar_by_vector( 
            vect, restrict_vocab=self.restrict_vocab))
        return set(sense_neighbors)

    def log_prec(self):
        if not self.test_size % 1000 and self.test_size:
            logging.info(
                'prec after testing on {} words: {:%}, self.good_ambig: {}'.format(
                    self.test_size, float(self.score_at10)/self.test_size, self.good_ambig))

if __name__ == '__main__':
    MultiSenseLinearTranslator(#target_embed=sys.argv[1]
                              ).main()

""" 
if len(good_vecs) > 2:
    sim_mx = good_vecs.dot(good_vecs.T)
    ij = np.unravel_index( np.argmin(sim_mx), sim_mx.shape)
    tgw1, tgw2 = [good_trans[i] for i in ij]
    logging.debug((sr_word, good_trans, tgw1, tgw2))
"""
