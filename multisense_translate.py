# coding: utf-8

import logging
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from gensim.models import KeyedVectors

class MultiSenseLinearTranslator():
    def __init__(self, source_mse, 
                 target_embed='/mnt/permanent/Language/English/Embed/GoogleNews-vectors-negative300.gensim', 
                 seed_filen='/mnt/store/makrai/data/language/hungarian/dict/wikt2dict-en-hu.by-freq'):
        logging.basicConfig(level=logging.INFO)
        logging.warning(
            'The order of the source and the target language in the arguments '
            'for embedding and in the words in the seed file is opposite')
        self.source_firsts = self.get_first_vectors(source_mse)
        self.target_embed = self.get_first_vectors(target_embed)
        logging.basicConfig(format="%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s", 
                            level=logging.INFO)
        self.seed_filen = seed_filen
        self.train_size = 5000
        self.test_size = 1000
        self.train_sr = np.zeros((self.train_size, self.source_firsts.syn0.shape[1]))
        self.train_tg = np.zeros((self.train_size, self.target_embed.syn0.shape[1]))

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
        with open(self.seed_filen) as seed_f:
            train_size = 0
            while train_size < self.train_size:
                tg, sr = seed_f.readline().strip().split()
                if sr in self.source_firsts and tg in self.target_embed:
                    self.train_sr[train_size] = self.source_firsts[sr]
                    self.train_tg[train_size] = self.target_embed[tg]
                    train_size += 1

            logging.info('trained on {} words'.format(len(self.train_sr)))
            regression = LinearRegression(n_jobs=-2)
            regression.fit(self.train_sr, self.train_tg)
            test_size = 0
            score_at10 = 0
            while test_size < self.test_size:
                line = seed_f.readline().strip()
                try:
                    tg, sr = line.split()
                except:
                    logging.err(line)
                if sr in self.source_firsts:
                    #if not test_size % 100:
                    #    logging.info('testing... ({} tested)'.format(test_size))
                    tg_vec = regression.coef_.dot(self.source_firsts[sr])
                    tg_10, _ = zip(*self.target_embed.wv.similar_by_vector(tg_vec, restrict_vocab=10000))
                    test_size += 1
                    if tg in tg_10:
                        score_at10 += 1
                        if not score_at10 % 100:
                            logging.info('prec@10 {:%}, test size {} {} {}'.format(float(score_at10)/test_size, test_size, sr, tg_10))
            prec_at10 = float(score_at10)/test_size
            logging.info('final prec@10 is {:%} on {} items'.format(prec_at10, test_size))
            return prec_at10

if __name__ == '__main__':
    print(MultiSenseLinearTranslator(source_mse=sys.argv[1]).main())
