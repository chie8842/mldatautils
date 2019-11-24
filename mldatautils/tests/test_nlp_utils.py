from mldatautils.nlp_utils import TokenFilter
from mldatautils.nlp_utils import Vectorizer
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import MeCab
import pytest

m = MeCab.Tagger()
parsed_sentence1 = m.parse('鍋に湯を沸かし、タピオカが透明になるまで茹で、（１５分から１７分位）ザルにあげて水で洗います。')
parsed_sentence2 = m.parse('茄子は乱切りし水にさらしてアク抜きする。')
parsed_sentences = [parsed_sentence1, parsed_sentence2]


def test_use_all():
    expected = [
        '鍋', 'に', '湯', 'を', '沸かす', '、', 'タピオカ', 'が',
        '透明', 'に', 'なる', 'まで', '茹でる', '、', '（', '１', '５', '分',
        'から', '１', '７', '分', '位', '）', 'ザル', 'に', 'あげる', 'て',
        '水', 'で', '洗う', 'ます', '。'
    ]
    actual = TokenFilter.use_all(parsed_sentence1)
    assert expected == actual

    # test use_jaconv
    expected = [
        '鍋', 'に', '湯', 'を', '沸かす', '、', 'タピオカ', 'が',
        '透明', 'に', 'なる', 'まで', '茹でる', '、', '(', '1', '5', '分',
        'から', '1', '7', '分', '位', ')', 'ザル', 'に', 'あげる', 'て',
        '水', 'で', '洗う', 'ます', '。'
    ]
    actual = TokenFilter.use_all(parsed_sentence1, use_jaconv=True)
    assert expected == actual

    # test pos_list
    expected = [
        '鍋', '湯', '沸かす', 'タピオカ',
        '透明', 'なる', '茹でる', '１', '５', '分',
        '１', '７', '分', '位', 'ザル', 'あげる',
        '水', '洗う',
    ]
    actual = TokenFilter.use_all(parsed_sentence1, pos_list=['名詞', '動詞'])
    assert expected == actual

    # except_words
    expected = [
        'に', '湯', 'を', '沸かす', '、', 'が',
        '透明', 'に', 'なる', 'まで', '茹でる', '、', '（', '１', '５', '分',
        'から', '１', '７', '分', '位', '）', 'ザル', 'に', 'あげる', 'て',
        '水', 'で', '洗う', 'ます', '。'
    ]
    actual = TokenFilter.use_all(parsed_sentence1, except_words=['鍋', 'タピオカ'])
    assert expected == actual

    # test exclude_numbers
    expected = [
        '鍋', 'に', '湯', 'を', '沸かす', '、', 'タピオカ', 'が',
        '透明', 'に', 'なる', 'まで', '茹でる', '、', '（', '分',
        'から', '分', '位', '）', 'ザル', 'に', 'あげる', 'て',
        '水', 'で', '洗う', 'ます', '。'
    ]
    actual = TokenFilter.use_all(parsed_sentence1, exclude_numbers=True)
    assert expected == actual

    # test exclude_symbols
    expected = [
        '鍋', 'に', '湯', 'を', '沸かす', 'タピオカ', 'が',
        '透明', 'に', 'なる', 'まで', '茹でる', '１', '５', '分',
        'から', '１', '７', '分', '位', 'ザル', 'に', 'あげる', 'て',
        '水', 'で', '洗う', 'ます'
    ]
    actual = TokenFilter.use_all(parsed_sentence1, exclude_symbols=True)
    assert expected == actual

    # test exclude_symbols
    expected = '会議するぞぉ'
    text = '会議するぞぉ~♡ ＼(^^)／'
    actual = TokenFilter.remove_special_characters(text)
    assert expected == actual


def test_vectorizer_fasttext():
    vectorizer = Vectorizer('fasttext', 'models/vectorizer/fasttext/fasttext_ipadic_8.model')
    word_list = TokenFilter.use_all(parsed_sentence1)
    words_list = [TokenFilter.use_all(parsed_sentence1), TokenFilter.use_all(parsed_sentence2)]

    assert type(vectorizer.model) == FastText
    assert vectorizer.model_dim == 8

    actual = vectorizer.transform_sentence(word_list)
    assert actual.shape == (8,)

    actual = vectorizer.transform_sentences(words_list, 8)
    assert actual.shape == (2, 8)


def test_vectorizer_word2vec():
    vectorizer = Vectorizer('word2vec', 'models/vectorizer/word2vec/word2vec_ipadic_8.model')
    word_list = TokenFilter.use_all(parsed_sentence1)
    words_list = [TokenFilter.use_all(parsed_sentence1), TokenFilter.use_all(parsed_sentence2)]

    assert type(vectorizer.model) == Word2Vec
    assert vectorizer.model_dim == 8

    actual = vectorizer.transform_sentence(word_list)
    assert actual.shape == (8,)

    actual = vectorizer.transform_sentences(words_list, 8)
    assert actual.shape == (2, 8)


def test_vectorizer_tfidf():
    vectorizer = Vectorizer('tfidf', 'models/vectorizer/tfidf/tfidf_ipadic_30000.model')
    word_list = TokenFilter.use_all(parsed_sentence1)
    words_list = [TokenFilter.use_all(parsed_sentence1), TokenFilter.use_all(parsed_sentence2)]

    assert type(vectorizer.model) == TfidfVectorizer
    assert vectorizer.model_dim <= 30000

    actual = vectorizer.transform_sentence(word_list)
    assert actual.shape == (vectorizer.model_dim,)

    actual = vectorizer.transform_sentences(words_list, 8)
    assert actual.shape == (2, 2)


def test_decompose():
    words_list = [TokenFilter.use_all(parsed_sentence1), TokenFilter.use_all(parsed_sentence2)]
    for i in range(10):
        words_list = words_list + words_list
    assert len(words_list) == 2048

    vectorizer = Vectorizer('tfidf', 'models/vectorizer/tfidf/tfidf_ipadic_30000.model')
    actual = vectorizer.transform_sentences(words_list, 100)
    assert actual.shape == (2048, 100)


def test_fail_transform_sentences():
    vectorizer = Vectorizer('fasttext', 'models/vectorizer/fasttext/fasttext_ipadic_8.model')
    words_list = [TokenFilter.use_all(parsed_sentence1), TokenFilter.use_all(parsed_sentence2)]

    actual = vectorizer.transform_sentences(words_list, 16)
    assert actual is None
