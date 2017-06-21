# coding: utf-8

import argparse
from collections import defaultdict
from functools import reduce
import logging
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from gensim.models import KeyedVectors


class MultiSenseLinearTranslator():
    """
    Cross-lingual word sense induction experiments: linear translation (Mikolov
    2013: Exploiting...) from multi-sense word embeddings (MSEs)
    """
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--source_mse',
            default='/mnt/permanent/Language/Hungarian/Embed/multiprot/adagram/mnsz/adagram-mnsz-600d-a.05-5p-m100_sense.mse')
        parser.add_argument(
            '--target_embed',
            default='/mnt/permanent/Language/English/Embed/glove.840B.300d.gensim')
        parser.add_argument(
            '--seed_dict',
            default='/mnt/store/makrai/data/language/hungarian/dict/wikt2dict-en-hu.by-freq',
            help='Name of the seed dictionary file. The order of the source'
            ' and the target language in the arguments for embedding and in'
            ' the words in the seed file is opposite')
        return parser.parse_args()

    def __init__(self):
        args = self.parse_args()
        self.source_mse_filen = args.source_mse
        self.source_firsts = self.get_first_vectors(args.source_mse)
        logging.basicConfig(
            format="%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s",
            level=logging.DEBUG)
        self.target_embed = self.get_first_vectors(args.target_embed)
        self.seed_filen = args.seed_dict

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
            self.train()
            return self.test()

    def train(self):
        self.train_size = 5000
        self.train_sr = np.zeros((self.train_size, self.source_firsts.syn0.shape[1]))
        self.train_tg = np.zeros((self.train_size, self.target_embed.syn0.shape[1]))
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
        self.test_size_goal = 1000
        self.restrict_vocab = 10000
        self.test_dict = self.read_test_dict()
        with open(self.source_mse_filen) as source_mse:
            logging.info(
                'skipping header: {}'.format(source_mse.readline().strip()))
            self.test_size_act = 0
            self.score_at10 = 0
            self.good_ambig = 0
            sr_word = ''
            sr_vecs = []
            while self.test_size_act < self.test_size_goal:
                line = source_mse.readline()
                if not line:
                    logging.warning(
                        'tested on {} items'.format(self.test_size_act))
                    break
                new_sr_word, vect_str = line.strip().split(maxsplit=1)
                if new_sr_word != sr_word:
                    if sr_word in self.test_dict:
                        self.eval_word(sr_word, sr_vecs)
                    sr_word = new_sr_word
                    sr_vecs = []
                sr_vecs.append(np.fromstring(vect_str, sep=' ').reshape((1,-1)))
        return float(self.score_at10)/self.test_size_act

    def read_test_dict(self):
        with open(self.seed_filen) as seed_f:
            self.test_dict = defaultdict(set)
            for line in seed_f.readlines():
                tg, sr = line.split()
                self.test_dict[sr].add(tg)
            return self.test_dict

    def eval_word(self, sr_word, sr_vecs):
        self.log_prec()
        self.test_size_act += 1
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
        if not self.test_size_act % 1000 and self.test_size_act:
            logging.info(
                'prec after testing on {} words: {:%}, self.good_ambig: {}'.format(
                    self.test_size_act, float(self.score_at10)/self.test_size_act, self.good_ambig))


if __name__ == '__main__':
    print(MultiSenseLinearTranslator().main())

"""
if len(good_vecs) > 2:
    sim_mx = good_vecs.dot(good_vecs.T)
    ij = np.unravel_index( np.argmin(sim_mx), sim_mx.shape)
    tgw1, tgw2 = [good_trans[i] for i in ij]
    logging.debug((sr_word, good_trans, tgw1, tgw2))
    """
