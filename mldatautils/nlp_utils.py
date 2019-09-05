import gensim
from gensim.models import doc2vec
from gensim.models import word2vec
from gensim.models import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import jaconv
from logger import logger_config

logger = logger_config('nlp_utils')


def extract_all(parsed_sentence, tokenizer='mecab', use_jaconv=False):
    '''
    Extract lemmatized? tokens from tokenized sentence.
    Currently this function suports only MeCab default format as input.
    Args:
        parsed_sentence(str): output of MeCab.Tagger().parse(sentence)
    Returns:
        word_list(list): tokens
    '''

    word_infos = parsed_sentence.split('\n')
    word_list = _extract_lemmatized_words(word_infos, use_jaconv)
    return word_list


def _extract_lemmatized_words(word_infos, use_jaconv):
    word_list = []
    for word_info in word_infos:
        word = _extract_lemmatized_word(word_info, use_jaconv)
        if word not in ('', 'EOS'):
            word_list.append(word)
    return word_list


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


def extract_noun(parsed_sentence, tokenizer='mecab', use_jaconv=False):
    '''
    Extract lemmatized? tokens from tokenized **noun** sentence.
    Args:
        parsed_sentence(str): output of MeCab.Tagger().parse(sentence)
    Returns:
        word_list(list): tokens
    '''
    word_infos = parsed_sentence.split('\n')
    word_list = _extract_noun_words(word_infos, use_jaconv)
    return word_list


def _extract_noun_words(word_infos, use_jaconv):
    word_list = []
    for word_info in word_infos:
        word = _extract_noun_word(word_info, use_jaconv)
        if word is not None:
            word_list.append(word)
    return word_list


def _extract_noun_word(word_info, use_jaconv):
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


def load_model(model_type, model_file):
    if model_type == 'doc2vec':
        model = doc2vec.Doc2Vec.load(model_file)
        vector_size = model.vector_size
    elif model_type == 'word2vec':
        model = word2vec.Word2Vec.load(model_file)
        vector_size = model.vector_size
    elif model_type == 'fasttext':
        model = fasttext.FastText.load(model_file)
        vector_size = model.vector_size
    else:  # tfidf
        with open(model_file, mode='rb') as f:
            model = pickle.load(f)
            vector_size = len(model.vocabulary_)
    return model, vector_size


def sentences_vectorizer(parsed_sentences, selector, use_jaconv, model, size, vector_size):
    if size < vector_size:
        logger.error('vector size should be bigger than size.')
        return None
    vector = np.zeros((len(parsed_sentences), size))
    for i, parsed_sentence in enumerate(parsed_sentences):
        vector[i] = sentence_vectorizer(parsed_sentence, selector, use_jaconv, model, size)
    vector = vector.astype(np.float32)
    if size > vector_size:
        logger.info('decompose vector with TruncatedSVD')
        vector = TruncatedSVD(n_components=vector_size).fit_transform(vector)
    return vector


def sentence_vectorizer(parsed_sentence, selector, use_jaconv, model, vector_size):
    """
    Args:
        parsed_sentence(str): output of MeCab.Tagger().parse(sentence)
        selector(func): extract_all(use all word) or extract_noun(use only noun)
        use_jaconv(bool): use jaconv(Japanese character interconverter) or not
        model: model of `word2vec` or `doc2vec` or `fasttext` or `tfidf`
    size: the vector size of vectorizer model
        vector_size: the size of result vector 
    """
    vector = np.zeros(vector_size)
    word_num = 0
    word_list = selector(parsed_sentence, use_jaconv=use_jaconv)
    logger.info('tokens: {}'.format(word_list))

    if type(model) == TfidfVectorizer:
        if len(model.vocaburary_) != vector_size:
            logger.error(f'model.vector_size is not equal to {vector_size}')
            return vector
        vector = model.transform(' '.join(word_list)).toarray().astype(np.float32)
    elif type(model) == gensim.models.doc2vec.Doc2Vec:
        if model.vector_size != vector_size:
            logger.error(f'model.vector_size is not equal to {vector_size}')
            return vector
        vector = model.infer_vector(word_list)
    else: # word2vec or fasttext
        if model.vector_size != vector_size:
            logger.error(f'model.vector_size is not equal to {vector_size}')
            return vector
        for word in word_list:
            try:
                word_vector = model.wv[word]
                vector = vector + word_vector
                word_num += 1
            except:
                logger.info('{} isn\'t in vocab'.format(word))
    if word_num != 0:
        vector = vector / word_num
    return vector
