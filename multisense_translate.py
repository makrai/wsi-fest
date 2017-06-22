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
            default='/mnt/permanent/Language/Hungarian/Embed/multiprot/adagram/mnsz/'
            'adagram-mnsz-600d-a.05-5p-m100_sense.mse')
        parser.add_argument(
            '--target_embed',
            default='/mnt/permanent/Language/English/Embed/glove.840B.300d.gensim')
        parser.add_argument(
            '--seed_dict',
            default='/mnt/store/makrai/data/language/hungarian/dict/' 
            'wikt2dict-en-hu.by-freq',
            help='Name of the seed dictionary file. The order of the source'
            ' and the target language in the arguments for embedding and in'
            ' the words in the seed file is opposite')
        parser.add_argument(
            '--general-linear-mapping', dest='orthog', action='store_false')
        parser.add_argument('--translate_all', action='store_true')
        parser.add_argument('--reverse', action='store_true')
        return parser.parse_args()

    def __init__(self):
        def get_first_vectors(filen):
            root, ext = os.path.splitext(filen)
            gens_fn = '{}.gensim'.format(root)
            if os.path.isfile(gens_fn):
                return KeyedVectors.load(gens_fn)
            else:
                embed = KeyedVectors.load_word2vec_format(filen)
                embed.save(gens_fn)
                return embed

        self.args = self.parse_args()
        self.source_firsts = get_first_vectors(self.args.source_mse)
        logging.basicConfig(
            format="%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s",
            level=logging.DEBUG)
        self.target_embed = get_first_vectors(self.args.target_embed)


    def main(self):
        with open(self.args.seed_dict) as self.seed_f:
            self.train()
            return self.test()

    def train(self, train_size_goal = 5000):
        self.train_sr = np.zeros((train_size_goal, self.source_firsts.syn0.shape[1]))
        self.train_tg = np.zeros((train_size_goal, self.target_embed.syn0.shape[1]))
        train_size_act = 0
        while train_size_act < train_size_goal:
            tg, sr = self.seed_f.readline().strip().split()
            if sr in self.source_firsts and tg in self.target_embed:
                self.train_sr[train_size_act] = self.source_firsts[sr]
                self.train_tg[train_size_act] = self.target_embed[tg]
                train_size_act += 1
        logging.info('trained on {} words'.format(len(self.train_sr)))
        if self.args.orthog:
            # formula taken from https://github.com/hlt-bme-hu/eval-embed
            m = self.train_sr.T.dot(self.train_tg)
            u, _, v = np.linalg.svd(m, full_matrices=True)
            s = np.zeros((v.shape[0], u.shape[1]))
            np.fill_diagonal(s, 1)
            self.regression = LinearRegression()
            self.regression.coef_ = v.T.dot(s).dot(u.T)
        else:
            self.regression = LinearRegression(n_jobs=-2)
            self.regression.fit(self.train_sr, self.train_tg)

    def test(self, restrict_vocab=10000):
        def read_test_dict():
            with open(self.args.seed_dict) as seed_f:
                self.test_dict = defaultdict(set)
                for line in seed_f.readlines():
                    tg, sr = line.split()
                    self.test_dict[sr].add(tg)

        def neighbors_by_vector(vect):
            sense_neighbors, _ = zip(*self.target_embed.similar_by_vector(
                vect, restrict_vocab=self.restrict_vocab))
            return set(sense_neighbors)

        def eval_word(sr_word, sr_vecs):
            # TODO monitor the neighborhood rank of the good translations
            self.test_size_act += 1
            tg_vecs = np.concatenate(sr_vecs).dot(self.regression.coef_.T)
            neighbor_by_vec = [neighbors_by_vector(v) for v in tg_vecs]
            if not self.test_size_act % 100:
                logging.debug((sr_word, neighbor_by_vec))
            hit_by_vec = [ns.intersection(self.test_dict[sr_word]) 
                          for ns in neighbor_by_vec]
            if reduce(set.union, hit_by_vec):
                self.score_at10 += 1
                common_hits = reduce(set.intersection, [n for n in hit_by_vec if n])
                uniq_hits_by_vec = [trans - common_hits for trans in hit_by_vec]
                uniq_hit_sets = set(' '.join(s) for s in uniq_hits_by_vec if s)
                uniq_hit_sets = [neigh.split(' ') for neigh in uniq_hit_sets]
                uniq_hit_sets.sort(key=len, reverse=True)
                if len(uniq_hit_sets) > 1:
                    self.good_disambig += 1
                    #if len(uniq_hit_sets) ==2:
                    logging.debug(( sr_word, uniq_hit_sets,
                                   '_'.join(common_hits), self.good_disambig))
            self.log_prec()

        read_test_dict()
        logging.info('Testing...')
        self.test_size_goal = 1000
        self.test_size_act = 0
        self.score_at10 = 0
        if self.args.reverse:
            return self.reverse_nn()
        self.restrict_vocab = restrict_vocab
        with open(self.args.source_mse) as source_mse:
            logging.info(
                'skipping header: {}'.format(source_mse.readline().strip()))
            self.good_disambig = 0
            sr_word = ''
            sr_vecs = []
            while self.args.translate_all or self.test_size_act < self.test_size_goal:
                line = source_mse.readline()
                if not line:
                    logging.info(
                        'tested on {} items'.format(self.test_size_act))
                    break
                new_sr_word, vect_str = line.strip().split(maxsplit=1)
                if new_sr_word != sr_word:
                    if sr_word in self.test_dict:
                        eval_word(sr_word, sr_vecs)
                    sr_word = new_sr_word
                    sr_vecs = []
                sr_vecs.append(np.fromstring(vect_str, sep=' ').reshape((1,-1)))
        return float(self.score_at10)/self.test_size_act

    def log_prec(self):
        if not self.test_size_act % 1000 and self.test_size_act:
            logging.info(
                'prec after testing on {} words: {:%} (good_disambig: {})'.format(
                    self.test_size_act,
                    float(self.score_at10)/self.test_size_act,
                    self.good_disambig))

    def reverse_nn(self, prec_level=10, vocab_limit = 50000):
        """
        Instead of the nearest neighbors of the computed point in target space,
        we rank source words by the distance of their translations to each
        target word. Source words are translated to the target word for which
        they have the lowest neighbor rank, following Dinu et al (2015).

        G. Dinu, A. Lazaridou and M. Baroni
        Improving zero-shot learning by mitigating the hubness problem.
        Proceedings of ICLR 2015, workshop track
        """ 
        def read_sr_eembed():
            vocab_size, dim = open(self.args.source_mse).readline().strip().split()
            logging.info('Reading source mx...')
            source_mse = np.genfromtxt(
                self.args.source_mse, skip_header=1, max_rows=vocab_limit,
                usecols=np.arange(1, int(dim)+1), dtype='float16', comments=None)
            sr_vocab = [line.split()[0] for line in
                        open(self.args.source_mse).readlines()[:vocab_limit]]
            logging.debug('Source vocab and mx read {}'.format(
                (len(sr_vocab), source_mse.shape)))
            return sr_vocab, source_mse

        def get_rev_rank():
            logging.info('Populating reverse neighbor rank mx...')
            rev_rank_col_blocks = [] 
            batch_size = 10000
            n_batch = int(min(self.target_embed.syn0.shape[0], vocab_limit) /
                          batch_size)
            for i in range(n_batch):
                tg_batch = self.target_embed.syn0[i*batch_size:(i+1)*batch_size]
                block = np.argsort(np.argsort(-translated_points.dot(tg_batch.T),
                                   axis=0).astype('int16'),
                                   axis=0).astype('int16')
                #32768
                rev_rank_col_blocks.append(block)
                logging.debug( 
                    '{:.1%} of reverse neighbor rank mx populated, {} {}'.format(
                        float(i+1)/n_batch, block.dtype, block.shape))
            return np.concatenate(rev_rank_col_blocks, axis=1)

        sr_vocab, source_mse = read_sr_eembed()
        translated_points = source_mse.dot(self.regression.coef_.T)
        rev_rank_mx = get_rev_rank()
        test_size_act = 0
        score_at10 = 0
        for sr_word, rev_rank_row in zip(sr_vocab, rev_rank_mx):
            if sr_word in self.test_dict:
                test_size_act += 1
                tg_words = [self.target_embed.index2word[i] 
                            for  i in np.argsort(rev_rank_row)[:prec_level]]
                if not test_size_act % 100:
                    logging.debug((sr_word, tg_words))
                if self.test_dict[sr_word].intersection(set(tg_words)):
                    score_at10 += 1
                if test_size_act == self.test_size_goal:
                    return float(self.score_at10)/test_size_act


if __name__ == '__main__':
    print(MultiSenseLinearTranslator().main())

"""
if len(good_vecs) > 2:
    sim_mx = good_vecs.dot(good_vecs.T)
    ij = np.unravel_index( np.argmin(sim_mx), sim_mx.shape)
    tgw1, tgw2 = [hit_by_vec[i] for i in ij]
    logging.debug((sr_word, hit_by_vec, tgw1, tgw2))
    """
