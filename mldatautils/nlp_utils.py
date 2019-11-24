import gensim
from gensim.models import doc2vec
from gensim.models import word2vec
from gensim.models import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import jaconv
import pickle
import re
from mldatautils.logger import logger_config


class TokenFilter(object):
    @staticmethod
    def use_all_for_sentences(
        parsed_sentences,
        use_jaconv=False,
        pos_list=[],
        except_words=[],
        exclude_numbers=False,
        exclude_symbols=False
    ):
        '''
        Extract lemmatized? tokens from tokenized sentence.
        Currently this function suports only MeCab default format as input.
        Args:
            parsed_sentences(list): list of output of MeCab.Tagger().parse(sentence)
            use_jaconv(bool): normalize tokens with jaconv or not
            pos_list(list): if we specify `pos_list=['名詞', '動詞']`,
                            it returns only the token which is `名詞` or `動詞`.
            except_words(list): if we specify `except_words=['基本', 'きほん']`,
                                it returns token list without word '基本' and 'きほん'.
            exclude_numbers(boot): if we specify `True`, it returns token list
                                   without `名詞 数` such as `2019`.
            exclude_symbols(bool): if we specify `True`, it returns token list
                                   without `記号`.
        Returns:
            word_list(list): tokens
        '''
        word_lists = [[]] * len(parsed_sentences)
        for i, parsed_sentence in enumerate(parsed_sentences):
            word_lists[i] = TokenFilter.use_all(
                parsed_sentence, use_jaconv, pos_list, except_words, exclude_numbers, exclude_symbols)
        return word_lists

    @staticmethod
    def use_all(
        parsed_sentence,
        use_jaconv=False,
        pos_list=[],
        except_words=[],
        exclude_numbers=False,
        exclude_symbols=False
    ):
        '''
        Extract lemmatized? tokens from tokenized sentence.
        Currently this function suports only MeCab default format as input.
        Args:
            parsed_sentence(str): output of MeCab.Tagger().parse(sentence)
            use_jaconv(bool): normalize tokens with jaconv or not
            pos_list(list): if we specify `pos_list=['名詞', '動詞']`,
                            it returns only the token which is `名詞` or `動詞`.
            except_words(list): if we specify `except_words=['基本', 'きほん']`,
                                it returns token list without word '基本' and 'きほん'.
            exclude_numbers(boot): if we specify `True`, it returns token list
                                   without `名詞 数` such as `2019`.
            exclude_symbols(bool): if we specify `True`, it returns token list
                                   without `記号`.
        Returns:
            word_list(list): tokens
        '''

        word_infos = parsed_sentence.split('\n')
        word_list = TokenFilter._extract_lemmatized_words(
            word_infos,
            use_jaconv,
            pos_list,
            except_words,
            exclude_numbers,
            exclude_symbols)
        return word_list

    @staticmethod
    def _extract_lemmatized_words(
        word_infos,
        use_jaconv=False,
        pos_list=[],
        except_words=[],
        exclude_numbers=False,
        exclude_symbols=False
    ):
        word_list = []
        for word_info in word_infos:
            word = TokenFilter._extract_lemmatized_word(
                word_info,
                use_jaconv,
                pos_list,
                exclude_numbers,
                exclude_symbols)
            except_words = ['', 'EOS'] + except_words
            if word not in except_words:
                word_list.append(word)
        return word_list

    @staticmethod
    def _extract_lemmatized_word(
        word_info,
        use_jaconv=False,
        pos_list=[],
        exclude_numbers=False,
        exclude_symbols=False
    ):
        word_info = word_info.split('\t')
        if len(word_info) > 1:
            word_details = word_info[1].split(',')
            if pos_list != [] and word_details[0] not in pos_list:
                return ''
            if exclude_numbers and word_details[1] == '数':
                return ''
            if exclude_symbols and word_details[0] == '記号':
                return ''
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
    def remove_special_characters(text):
        non_CJK_patterns = re.compile(
            "[^"
            u"\U00003040-\U0000309F"  # Hiragana
            u"\U000030A0-\U000030FF"  # Katakana
            u"\U0000FF65-\U0000FF9F"  # Half width Katakana
            u"\U0000FF10-\U0000FF19"  # Full width digits
            u"\U0000FF21-\U0000FF3A"  # Full width Upper case  English Alphabets
            u"\U0000FF41-\U0000FF5A"  # Full width Lower case English Alphabets
            u"\U00000030-\U00000039"  # Half width digits
            u"\U00000041-\U0000005A"  # Half width  Upper case English Alphabets
            u"\U00000061-\U0000007A"  # Half width Lower case English Alphabets
            u"\U00003190-\U0000319F"  # Kanbun
            u"\U00004E00-\U00009FFF"  # CJK unified ideographs. kanjis
            "]+", flags=re.UNICODE)
        return non_CJK_patterns.sub(r"", text)


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

    def transform_sentences(self, words_list, vector_dim=None, pooling='avg'):
        """
        Args:
            words_list(str): a list or pd.Series of word list such as [['今日', 'は', 'いい', '天気'],[...]]
            vector_dim(int): dimension of output vector
            pooling(str): 'avg' or 'max'. vector summalize method when we use `word2vec` or `fasttext`
                          for sentence vectorizer
        """
        if vector_dim is None:
            vector_dim = self.model_dim
        if self.model_dim < vector_dim:
            self.logger.error('vector size should be bigger than size.')
            return None

        # TODO: transform sentences in one time when use doc2vec
        if self.model_type == 'tfidf':
            words_list = [' '.join(word_list) for word_list in words_list]
            vectors = self.model.transform(words_list)
        else:
            vectors = np.zeros((len(words_list), self.model_dim), dtype='float32')
            for i, word_list in enumerate(words_list):
                vectors[i] = self.transform_sentence(word_list, pooling='avg')
        if self.model_dim > vector_dim:
            self.logger.info('decompose vector with TruncatedSVD')
            vectors = TruncatedSVD(n_components=vector_dim).fit_transform(vectors)
        return vectors

    def transform_sentence(self, word_list, pooling='avg'):
        """
        Args:
            word_list(str): a list or pd.Series of words such as ['今日', 'は', 'いい', '天気']
            vector_dim(int): dimension of output vector
            pooling(str): 'avg' or 'max'. vector summalize method when we use `word2vec` or `fasttext`
                          for sentence vectorizer
        """
        vector = np.zeros(self.model_dim, dtype='float32')
        word_num = 0
        self.logger.info('tokens: {}'.format(word_list))
        if not word_list:
            return vector

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
