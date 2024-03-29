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
                 seed_dict=None, orthog=False, translate_all=False, reverse=True,
                 restrict_vocab=2**15, prec_level=10):

        def get_first_vectors(filen):
            root, ext = os.path.splitext(filen)
            gens_fn = '{}.gensim'.format(root)
            if os.path.isfile(gens_fn):
                embed = KeyedVectors.load(gens_fn)
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
            self.args.source_mse = None
            self.args.target_embed = None
            self.args.seed_dict = None
            default_proxy = get_proxy(default_config_filen)
        if not self.args.source_mse:
            self.args.source_mse = source_mse if source_mse else default_proxy['SourceMse']
        if not self.args.target_embed:
            self.args.target_embed = target_embed if target_embed else default_proxy['TargetEmbed']
        if not self.args.seed_dict:
            self.args.seed_dict = seed_dict if seed_dict else default_proxy['SeedDict']

        # This first command is not logged because reading in MSEs with gensim
        # causes many warnings
        self.source_firsts = get_first_vectors(self.args.source_mse)
        logging.basicConfig(
            format="%(asctime)s %(module)s (%(lineno)s) %(levelname)s %(message)s",
            level=logging.DEBUG)
        self.target_embed = get_first_vectors(self.args.target_embed)

    def main(self):
        self.norm_cent()
        with open(self.args.seed_dict) as self.seed_f:
            self.train()
            return self.test()

    def norm_cent(self):
        logging.debug(self.args.cent_norm)
        if self.args.cent_norm in ['cent', 'cent_norm']:
            self.sr_center = self.get_center(self.source_firsts.syn0)
            self.tg_center = self.get_center(self.target_embed.syn0)
            self.source_firsts.syn0 -= self.sr_center
            self.target_embed.syn0 -= self.tg_center 
        if self.args.cent_norm in ['norm', 'cent_norm', 'norm_cent']:
            self.normalize(self.source_firsts.syn0)
            self.normalize(self.target_embed.syn0)
        if self.args.cent_norm == 'norm_cent':
            self.sr_center = self.get_center(self.source_firsts.syn0)
            self.tg_center = self.get_center(self.target_embed.syn0)
            self.source_firsts.syn0 -= self.sr_center
            self.target_embed.syn0 -= self.tg_center 

    def normalize(self, embed):
        embed /= np.apply_along_axis(np.linalg.norm, 1, embed).reshape((-1,1))

    def get_center(self, embed):
        return np.sum(embed, axis=0)/embed.shape[0]

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
        def init_test():
            self.test_size_goal = 1000
            self.test_size_act = 0
            self.score = 0
            self.gold_ambig = 0
            read_test_dict()
            if self.args.reverse:
                self.sr_i = 0
                sr_vocab, source_mse = read_sr_embed()
                self.translated_points = source_mse.dot(self.regression.coef_.T)
                self.normalize(self.target_embed.syn0)
                self.normalize(self.translated_points)
                self.rev_rank_mx = get_rev_rank()

        def read_test_dict():
            self.test_dict = defaultdict(set)
            for line in self.seed_f.readlines():
                tg, sr = line.split()
                self.test_dict[sr].add(tg)

        def log_prec():
            logging.info(
                'prec after testing on {} words: {:.3%} {:.3%}'.format(
                    self.test_size_act,
                    self.score/self.test_size_act,
                    self.gold_ambig/self.sys_ambig))

        def eval_word(sr_word, neighbor_by_vec):
            hit_by_vec = [ns.intersection(self.test_dict[sr_word])
                          for ns in neighbor_by_vec]
            good_trans = reduce(set.union, hit_by_vec)
            if good_trans:
                self.score += 1
                uniq_hit_sets = set(' '.join(hs) for hs in hit_by_vec if hs)
                if len(uniq_hit_sets) > 1:
                    self.gold_ambig += 1
                    uniq_hit_sets = [neigh.split()
                                     for neigh in uniq_hit_sets]
                    uniq_hit_sets.sort(key=len, reverse=True)
                    #if len(uniq_hit_sets) > 2 and self.args.prec_level != 1:
                    # When there are sense vectors with more than two hits, the
                    # choice of the corresponding target  words is arbitrary.)
                    #inds = [self.target_embed.word2index(wt[0]) 
                    #for wt in uniq_hit_sets]
                    #vecs = self.target_embed.syn0[inds]
                    #logging.debug(vecs.dot(vecs.T))
                    w1, w2 = [list(hits)[0] for hits in uniq_hit_sets[:2]]
                    sim = self.target_embed.similarity(w1, w2)
                    self.sims.append(sim)
                    msg = '{:.4}\t{}\t{}\t{:.2}'.format(
                        sim, sr_word, uniq_hit_sets,
                        len(good_trans)/len(self.test_dict[sr_word]))
                    if self.args.verbose:
                        print(msg)
                    logging.debug(msg)
            if not self.test_size_act % 1000 and self.test_size_act:
                log_prec()

        # Inner functions used only in the fwd (non-reverse) version
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
            if self.args.cent_norm in ['cent', 'cent_norm']:
                source_mse -= self.sr_center
            if self.args.cent_norm in ['norm', 'cent_norm', 'norm_cent']:
                self.normalize(source_mse)
            if self.args.cent_norm == 'norm_cent':
                source_mse -= self.sr_center
            logging.info(
                'Source vocab and mx read {}'.format(source_mse.shape))
            return sr_vocab, source_mse

        def get_rev_rank(debug=False, break_ties=False):
            if break_ties:
                raise NotImplementedError
                sim_mx = self.translated_points.dot(
                    self.target_embed.syn0[:self.args.restrict_vocab].T).astype(
                        'float16')
                fwd_ranks = (-sim_mx).argsort(axis=0).astype('uint16')
                fwd_ranks = fwd_ranks.argsort(axis=0).astype('uint16')
                logging.debug(fwd_ranks[:,0])
                fwd_ranks = fwd_ranks - sim_mx
                # Not -=. because Cannot cast ufunc subtract output from
                # dtype('float64') to dtype('int64') with casting rule
                # 'same_kind'
                return fwd_ranks.argsort().astype('uint16')
            else:
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

        logging.info('Testing...')
        init_test()
        with open(self.args.source_mse) as source_mse:
            logging.debug(
                'skipping header: {}'.format(source_mse.readline().strip()))
            sr_word = ''
            sr_vecs = [] # sr_vecs is NOT used in reverse mode
            act_sense_indices = []
            self.sims = []
            self.sys_ambig = 0
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
                            if len(act_sense_indices) > 1:
                                self.sys_ambig += 1
                            rev_rank_row_block = self.rev_rank_mx[act_sense_indices]
                            neighbor_by_vec = [
                                set(self.target_embed.index2word[i]
                                 for  i in rev_rank_row[:self.args.prec_level])
                                for rev_rank_row in rev_rank_row_block]
                        else:
                            if len(sr_vecs) > 1:
                                self.sys_ambig += 1
                            tg_vecs = np.concatenate(sr_vecs).dot(self.regression.coef_.T)
                            if self.args.cent_norm in ['cent', 'cent_norm']:
                                tg_vecs -= self.tg_center
                            if self.args.cent_norm in ['norm', 'cent_norm', 'norm_cent']:
                                self.normalize(tg_vecs)
                            if self.args.cent_norm == 'norm_cent':
                                tg_vecs -= self.tg_center 
                            neighbor_by_vec = [neighbors_by_vector(v) for v in tg_vecs]
                        eval_word(sr_word, neighbor_by_vec)
                    sr_word = new_sr_word
                    sr_vecs = []
                if self.args.reverse:
                    act_sense_indices.append(self.sr_i)
                    self.sr_i += 1
                else:
                    sr_vecs.append(np.fromstring(vect_str, sep=' ').reshape((1,-1)))
            print('{:.1%} {:.2%}'.format(self.score/self.test_size_act,
                                         self.gold_ambig/self.sys_ambig))
        return self.sims


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='hlt_bp.ini')
    parser.add_argument('--source_mse')
    parser.add_argument('--target_embed')
    parser.add_argument(
        '--seed_dict',
        help='Name of the seed dictionary file. The order of the source and'
        ' the target language in the arguments for embeddings and in the word'
        ' pairs in the seed file is opposite')
    parser.add_argument( '--orthog', action='store_true')
    parser.add_argument('--translate_all', action='store_true')
    parser.add_argument(
        '--fwd-nn-search', dest='reverse', action='store_false',
        help='non-reverse NN search')
    parser.add_argument('--restrict_vocab', type=int, default=2**15)
    parser.add_argument('--prec_level', type=int, default=10)
    parser.add_argument('--non-verbose', dest='verbose', action='store_false')
    parser.add_argument(
        '--cent_norm', default='vanilla',
        choices=['vanilla', 'cent', 'norm', 'cent_norm', 'norm_cent'])
    return parser.parse_args()


if __name__ == '__main__':
    MultiSenseLinearTranslator(args=parse_args()).main()
