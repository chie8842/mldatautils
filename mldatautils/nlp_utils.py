import gensim
from gensim.models import doc2vec
from gensim.models import word2vec
from gensim.models import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import jaconv
import pickle
from mldatautils.logger import logger_config


class TokenFilter(object):
    """
    TODO: use_all and use_noun should be merged.
    """
    @staticmethod
    def use_all(parsed_sentence, tokenizer='mecab', use_jaconv=False):
        '''
        Extract lemmatized? tokens from tokenized sentence.
        Currently this function suports only MeCab default format as input.
        Args:
            parsed_sentence(str): output of MeCab.Tagger().parse(sentence)
        Returns:
            word_list(list): tokens
        '''

        word_infos = parsed_sentence.split('\n')
        word_list = TokenFilter._extract_lemmatized_words(word_infos, use_jaconv)
        return word_list

    @staticmethod
    def _extract_lemmatized_words(word_infos, use_jaconv):
        word_list = []
        for word_info in word_infos:
            word = TokenFilter._extract_lemmatized_word(word_info, use_jaconv)
            if word not in ('', 'EOS'):
                word_list.append(word)
        return word_list

    @staticmethod
    def _extract_lemmatized_word(word_info, use_jaconv):
        word_info = word_info.split('\t')
        if len(word_info) > 1:
            word_details = word_info[1].split(',')
            if len(word_details) > 6 and word_details[6] != '*':
                word = word_details[6]
            else:
                word = word_info[0]
        else:
            word = word_info[0]
        if use_jaconv:
            word = jaconv.z2h(word, digit=True, ascii=True)
            word = jaconv.normalize(word)
        return word

    @staticmethod
    def use_noun(parsed_sentence, tokenizer='mecab', use_jaconv=False):
        '''
        Extract lemmatized? tokens from tokenized **noun** sentence.
        Args:
            parsed_sentence(str): output of MeCab.Tagger().parse(sentence)
        Returns:
            word_list(list): tokens
        '''
        word_infos = parsed_sentence.split('\n')
        word_list = TokenFilter._extract_lemmatized_nouns(word_infos, use_jaconv)
        return word_list

    @staticmethod
    def _extract_lemmatized_nouns(word_infos, use_jaconv):
        word_list = []
        for word_info in word_infos:
            word = TokenFilter._extract_lemmatized_noun(word_info, use_jaconv)
            if word is not None:
                word_list.append(word)
        return word_list

    @staticmethod
    def _extract_lemmatized_noun(word_info, use_jaconv):
        word_info = word_info.split('\t')
        word = word_info[0]
        if len(word_info) == 2 and word_info[1].split(',')[0] == '名詞':
            word_details = word_info[1].split(',')
            if len(word_details) > 6 and word_details[6] != '*':
                word = word_details[6]
            else:
                word = word_info[0]
            if use_jaconv:
                word = jaconv.z2h(word, digit=True, ascii=True)
                word = jaconv.normalize(word)
            return word


class Vectorizer(object):
    def __init__(self, model_type, model_file, log_level='WARN'):
        """
        model_type(str): `word2vec` or `doc2vec` or `fasttext` or `tfidf`
        model_file(str): file path of the model
        log_level(str): log level
        """
        self.logger = logger_config('vectorizer', log_level)
        self.model, self.model_dim = self._load_model(model_type, model_file)
        self.model_type = model_type

    def _load_model(self, model_type, model_file):
        if model_type == 'doc2vec':
            model = doc2vec.Doc2Vec.load(model_file)
            model_dim = model.vector_size
        elif model_type == 'word2vec':
            model = word2vec.Word2Vec.load(model_file)
            model_dim = model.vector_size
        elif model_type == 'fasttext':
            model = fasttext.FastText.load(model_file)
            model_dim = model.vector_size
        else:  # tfidf
            with open(model_file, mode='rb') as f:
                model = pickle.load(f)
                model_dim = len(model.vocabulary_)
        return model, model_dim

    def transform_sentences(self, parsed_sentences, token_filter, use_jaconv, vector_dim, pooling='avg'):
        """
        Args:
            parsed_sentences(str): a list or pd.Series of output of MeCab.Tagger().parse(sentence)
            token_filter(func): TokenFilter.use_all(use all word) or TokenFilter.use_noun(use only noun)
            use_jaconv(bool): use jaconv(Japanese character interconverter) or not
        """
        if self.model_dim < vector_dim:
            self.logger.error('vector size should be bigger than size.')
            return None

        # TODO: transform sentences in one time when use doc2vec or tfidf
        vectors = np.zeros((len(parsed_sentences), self.model_dim))
        for i, parsed_sentence in enumerate(parsed_sentences):
            vectors[i] = self.transform_sentence(parsed_sentence, token_filter, use_jaconv, pooling='avg')
        vectors = vectors.astype(np.float32)
        if self.model_dim > vector_dim:
            self.logger.info('decompose vector with TruncatedSVD')
            vectors = TruncatedSVD(n_components=vector_dim).fit_transform(vectors)
        return vectors

    def transform_sentence(self, parsed_sentence, token_filter, use_jaconv, pooling='avg'):
        """
        Args:
            parsed_sentence(str): output of MeCab.Tagger().parse(sentence)
            token_filter(func): TokenFilter.use_all(use all word) or TokenFilter.use_noun(use only noun)
            use_jaconv(bool): use jaconv(Japanese character interconverter) or not
            pooling(str): avg or max
        """
        vector = np.zeros(self.model_dim)
        word_num = 0
        word_list = token_filter(parsed_sentence, use_jaconv=use_jaconv)
        self.logger.info('tokens: {}'.format(word_list))

        if self.model_type == 'tfidf':
            vector = self.model.transform([' '.join(word_list)]).toarray().astype(np.float32)[0]
        elif self.model_type == 'doc2vec':
            vector = model.infer_vector(word_list)
        elif self.model_type == 'fasttext':
            if pooling == 'avg':
                vector = self.model.wv[word_list].mean(axis=0)
            else:  # max
                vector = self.model.wv[word_list].max(axis=0)
        else:  # word2vec
            if pooling == 'avg':
                vector = self.model.wv[filter(lambda x: x in self.model.wv.vocab, word_list)].mean(axis=0)
            else:  # max
                vector = self.model.wv[filter(lambda x: x in self.model.wv.vocab, word_list)].max(axis=0)
        return vector
