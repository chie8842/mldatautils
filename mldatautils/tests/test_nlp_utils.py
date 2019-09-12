from datautils.nlp_utils import TokenFilter
from datautils.nlp_utils import Vectorizer
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


def test_use_noun():
    expected = ['鍋', '湯', 'タピオカ', '透明', '１', '５', '分', '１', '７', '分', '位', 'ザル', '水']
    actual = TokenFilter.use_noun(parsed_sentence1)
    assert expected == actual


def test_vectorizer_fasttext():
    vectorizer = Vectorizer('fasttext', 'models/vectorizer/fasttext/fasttext_ipadic_8.model')
    assert type(vectorizer.model) == FastText
    assert vectorizer.model_dim == 8

    actual = vectorizer.transform_sentence(parsed_sentence1, TokenFilter.use_all, True)
    assert actual.shape == (8,)

    actual = vectorizer.transform_sentences(parsed_sentences, TokenFilter.use_all, True, 8)
    assert actual.shape == (2, 8)


def test_vectorizer_word2vec():
    vectorizer = Vectorizer('word2vec', 'models/vectorizer/word2vec/word2vec_ipadic_8.model')
    assert type(vectorizer.model) == Word2Vec
    assert vectorizer.model_dim == 8

    actual = vectorizer.transform_sentence(parsed_sentence1, TokenFilter.use_all, True)
    assert actual.shape == (8,)

    actual = vectorizer.transform_sentences(parsed_sentences, TokenFilter.use_all, True, 8)
    assert actual.shape == (2, 8)


def test_vectorizer_tfidf():
    vectorizer = Vectorizer('tfidf', 'models/vectorizer/tfidf/tfidf_ipadic_30000.model')
    assert type(vectorizer.model) == TfidfVectorizer
    assert vectorizer.model_dim <= 30000

    actual = vectorizer.transform_sentence(parsed_sentence1, TokenFilter.use_all, True)
    assert actual.shape == (vectorizer.model_dim, )

    actual = vectorizer.transform_sentences(parsed_sentences, TokenFilter.use_all, True, vectorizer.model_dim)
    assert actual.shape == (2, vectorizer.model_dim)


def test_decompose():
    parsed_sentences_large = parsed_sentences
    for i in range(10):
        parsed_sentences_large = parsed_sentences_large + parsed_sentences_large
    assert len(parsed_sentences_large) == 2048

    vectorizer = Vectorizer('tfidf', 'models/vectorizer/tfidf/tfidf_ipadic_30000.model')
    actual = vectorizer.transform_sentences(parsed_sentences_large, TokenFilter.use_all, True, 100)
    assert actual.shape == (2048, 100)


def test_fail_transform_sentences():
    vectorizer = Vectorizer('fasttext', 'models/vectorizer/fasttext/fasttext_ipadic_8.model')
    with pytest.raises(Exception) as e_info:
       actual = vectorizer.transform_sentences(parsed_sentences_large, TokenFilter.use_all, True, 16)
