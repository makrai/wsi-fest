# coding: utf-8

import argparse
from collections import defaultdict
from functools import reduce
import logging
import os
import configparser

import numpy as np
from sklearn.linear_model import LinearRegression
from gensim.models import KeyedVectors

default_config_filen = 'hlt_bp.ini'

class MultiSenseLinearTranslator():
    """
    Cross-lingual word sense induction experiments: linear translation (Mikolov
    2013: Exploiting...) from multi-sense word embeddings (MSEs)
    """

    def __init__(self, args=None, source_mse=None, target_embed=None,
                 seed_dict=None, orthog=True, translate_all=False, reverse=True,
                 restrict_vocab=2**15, prec_level=10):
        def get_first_vectors(filen):
            root, ext = os.path.splitext(filen)
            gens_fn = '{}.gensim'.format(root)
            if os.path.isfile(gens_fn):
               return KeyedVectors.load(gens_fn)
            else:
                embed = KeyedVectors.load_word2vec_format(filen)
                embed.save(gens_fn)
                return embed

        def get_proxy(filen):
            config = configparser.ConfigParser()
            config.read(filen)
            return config['DEFAULT']

        if args:
            self.args = args
            default_proxy = get_proxy(args.config_file)
        else:
            self.args = argparse.Namespace()
            self.args.orthog = orthog
            self.args.translate_all = translate_all 
            self.args.reverse = reverse 
            self.args.restrict_vocab = restrict_vocab 
            self.args.prec_level = prec_level 
            default_proxy = get_proxy(default_config_filen)
        if not self.args.source_mse:
            self.args.source_mse = source_mse if source_mse else default_proxy['SourceMse']
        if not self.args.target_embed:
            self.args.target_embed = target_embed if target_embed else default_proxy['TargetEmbed']
        if not self.args.seed_dict:
            self.args.seed_dict = seed_dict if seed_dict else default_proxy['SeedDict']
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
        self.train_sr = np.zeros((train_size_goal,
                                  self.source_firsts.syn0.shape[1]))
        self.train_tg = np.zeros((train_size_goal,
                                  self.target_embed.syn0.shape[1]))
        train_size_act = 0
        while train_size_act < train_size_goal:
            tg, sr = self.seed_f.readline().strip().split()
            if sr in self.source_firsts and tg in self.target_embed:
                self.train_sr[train_size_act] = self.source_firsts[sr]
                self.train_tg[train_size_act] = self.target_embed[tg]
                train_size_act += 1
        logging.info('Trained on {} words'.format(len(self.train_sr)))
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

    def test(self):
        """
        In reverse mode, instead of ranking the nearest neighbors of the
        computed point in target space, we rank source words by the distance of
        their translations to each target word. Source words are translated to
        the target word for which they have the lowest neighbor rank, following
        Dinu et al (2015).

        G. Dinu, A. Lazaridou and M. Baroni
        Improving zero-shot learning by mitigating the hubness problem.
        Proceedings of ICLR 2015, workshop track
        """ 
        # Inner functions used both in reverse and normal mode
        def read_test_dict():
            self.test_dict = defaultdict(set)
            for line in self.seed_f.readlines():
                tg, sr = line.split()
                self.test_dict[sr].add(tg)

        def log_prec():
            logging.info(
                'prec after testing on {} words: {:%} (good_disambig: {})'.format(
                    self.test_size_act,
                    float(self.score)/self.test_size_act,
                    self.good_disambig))

        def eval_word(sr_word, neighbor_by_vec):
            hit_by_vec = [ns.intersection(self.test_dict[sr_word])
                          for ns in neighbor_by_vec]
            good_trans = reduce(set.union, hit_by_vec)
            if good_trans:
                self.score += 1
                common_hits = reduce(set.intersection, 
                                     [hits for hits in hit_by_vec if hits])
                uniq_hits_by_vec = [trans - common_hits
                                    for trans in hit_by_vec]
                uniq_hit_sets = set(' '.join(s) for s in uniq_hits_by_vec if s)
                if len(uniq_hit_sets) > 1:
                    self.good_disambig += 1
                    uniq_hit_sets = [neigh.split()
                                     for neigh in uniq_hit_sets]
                    uniq_hit_sets.sort(key=len, reverse=True)
                    w1, w2 = [list(hits)[0] for hits in uniq_hit_sets[:2]]
                    sim = self.target_embed.similarity(w1, w2)
                    self.sims.append(sim)
                    msg = '{} {} {} {} {:.2} {}'.format(
                        sim, sr_word, uniq_hit_sets, '_'.join(common_hits),
                        len(good_trans)/len(self.test_dict[sr_word]),
                        self.good_disambig)
                    if not self.args.silent:
                        print(msg)
                    #if not self.good_disambig % 10:
                    logging.debug(msg)
            if not self.test_size_act % 1000 and self.test_size_act:
                log_prec()

        # Inner functions used only in the vanilla (non-reverse) version
        def neighbors_by_vector(vect):
            sense_neighbors, _ = zip(*self.target_embed.similar_by_vector(
                vect, restrict_vocab=self.args.restrict_vocab, topn=self.args.prec_level))
            return set(sense_neighbors)
            # We loose the order of neighbors. Keeping it have been tried and
            # brought no improvement.

        # Inner functions used ony in reverse mode
        def read_sr_embed():
            logging.info(
                'Reading source mx from {}...'.format(self.args.source_mse))
            with open(self.args.source_mse) as source_mse:
                _, dim = source_mse.readline().strip().split()
                sr_vocab = [line.split()[0] for line in
                            source_mse.readlines()[:self.args.restrict_vocab]]
            source_mse = np.genfromtxt(
                self.args.source_mse, skip_header=1, max_rows=self.args.restrict_vocab,
                usecols=np.arange(1, int(dim)+1), dtype='float16',
                comments=None)
            logging.info(
                'Source vocab and mx read {}'.format(source_mse.shape))
            return sr_vocab, source_mse

        def get_rev_rank(debug=False):
            logging.info('Populating reverse neighbor rank mx...')
            rev_rank_col_blocks = []
            batch_size = 10000
            n_batch = max(int(min(self.target_embed.syn0.shape[0],
                                  self.args.restrict_vocab) / batch_size),
                          1)
            for i in range(n_batch):
                tg_batch = self.target_embed.syn0[i*batch_size:(i+1)*batch_size]
                block = self.translated_points.dot(tg_batch.T)
                block = (-block).argsort(axis=0).astype('uint16')
                block = block.argsort(axis=0).astype('uint16')
                rev_rank_col_blocks.append(block)
                logging.debug(
                    '{:.1%} of reverse neighbor rank mx populated, {} {}'.format(
                        float(i+1)/n_batch, block.dtype, block.shape))
            rev_rank_mx = np.concatenate(rev_rank_col_blocks,
                                         axis=1).astype('uint16')
            logging.debug('Min ranks: {}'.format(rev_rank_mx.min(axis=1)))
            rev_rank_mx = rev_rank_mx.argsort().astype('uint16')
            return rev_rank_mx

        def get_tie_broken_rev_rank():
            sim_mx = self.translated_points.dot(self.target_embed.syn0.T)
            # TODO

        def normalize(vecs):
            vecs /= np.apply_along_axis(np.linalg.norm, 1,
                                        vecs).reshape((-1,1))

        def init_test():
            self.test_size_goal = 1000
            self.test_size_act = 0
            self.score = 0
            self.good_disambig = 0
            read_test_dict()
            if self.args.reverse:
                if self.args.reverse:
                    self.sr_i = 0
                sr_vocab, source_mse = read_sr_embed()
                self.translated_points = source_mse.dot(self.regression.coef_.T)
                normalize(self.target_embed.syn0)
                normalize(self.translated_points)
                self.rev_rank_mx = get_rev_rank()

        logging.info('Testing...')
        init_test()
        with open(self.args.source_mse) as source_mse:
            logging.debug(
                'skipping header: {}'.format(source_mse.readline().strip()))
            sr_word = ''
            sr_vecs = [] # sr_vecs is NOT used in reverse mode
            act_sense_indices = []
            self.sims = []
            while (self.args.translate_all
                   or self.test_size_act < self.test_size_goal):
                line = source_mse.readline()
                if (self.args.reverse and self.sr_i > self.args.restrict_vocab
                    or not line):
                    logging.info(
                        'tested on {} items'.format(self.test_size_act))
                    break
                new_sr_word, vect_str = line.strip().split(maxsplit=1)
                if new_sr_word != sr_word:
                    if sr_word in self.test_dict:
                        self.test_size_act += 1
                        if self.args.reverse:
                            rev_rank_row_block = self.rev_rank_mx[act_sense_indices]
                            neighbor_by_vec = [
                                set(self.target_embed.index2word[i] 
                                 for  i in rev_rank_row[:self.args.prec_level]) 
                                for rev_rank_row in rev_rank_row_block]
                        else:
                            tg_vecs = np.concatenate(sr_vecs).dot(self.regression.coef_.T)
                            neighbor_by_vec = [neighbors_by_vector(v) for v in tg_vecs]
                        eval_word(sr_word, neighbor_by_vec)
                    sr_word = new_sr_word
                    sr_vecs = []
                if self.args.reverse:
                    act_sense_indices.append(self.sr_i)
                    self.sr_i += 1
                else:
                    sr_vecs.append(np.fromstring(vect_str, sep=' ').reshape((1,-1)))
                    # TODO normalize
        print('{:.1%} {}'.format(float(self.score)/self.test_size_act,
                                self.good_disambig))
        return self.sims


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file', default='hlt_bp.ini', 
        help='Name of the seed dictionary file. The order of the source and'
        ' the target language in the arguments for embeddings and in the word'
        ' pairs in the seed file is opposite') 
    parser.add_argument('--source_mse')
    parser.add_argument('--target_embed')
    parser.add_argument('--seed_dict')
    parser.add_argument(
        '--general-linear-mapping', dest='orthog', action='store_false')
    parser.add_argument('--translate_all', action='store_true')
    parser.add_argument('--vanilla-nn-search', dest='reverse',
                        action='store_false', 
                        help='Do not compute reverse NNs')
    parser.add_argument('--restrict_vocab', type=int, default=2**15)
    parser.add_argument('--prec_level', type=int, default=10)
    parser.add_argument('--silent', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    MultiSenseLinearTranslator(args=parse_args()).main()
