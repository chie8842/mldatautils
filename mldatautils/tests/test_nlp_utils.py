from mldatautils.tokenize_utils import extract_all, extract_noun
import MeCab

m = MeCab.Tagger()
parsed_sentence = m.parse("鍋に湯を沸かし、タピオカが透明になるまで茹で、（１５分から１７分位）ザルにあげて水で洗います。")

def test_extract_all():
    expected = ['鍋', 'に', '湯', 'を', '沸かす', '、', 'タピオカ', 'が', '透明', 'に', 'なる', 'まで', '茹でる', '、', '（', '１', '５', '分', 'から', '１', '７', '分', '位', '）', 'ザル', 'に', 'あげる', 'て', '水', 'で', '洗う', 'ます', '。', 'EOS', '']
    actual = extract_all(parsed_sentence)
    assert expected == actual

def test_extract_noun():
    expected = ['鍋', '湯', 'タピオカ', '透明', '１', '５', '分', '１', '７', '分', '位', 'ザル', '水']
    actual = extract_noun(parsed_sentence)
    assert expected == actual

