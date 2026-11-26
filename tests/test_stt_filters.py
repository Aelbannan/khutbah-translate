from khutbah.stt import collapse_repeats


def test_collapse_phrase_and_word_loops():
    assert collapse_repeats("a b c a b c a b c d")[0] == "a b c d"
    assert collapse_repeats("x مِنْسَاءَاتِ مِنْسَاءَاتِ مِنْسَاءَاتِ مِنْسَاءَاتِ y")[0] == "x مِنْسَاءَاتِ y"


def test_emphasis_kept():
    for t in ["no no, I said no no", "no one can ever ever imagine", "then then"]:
        assert collapse_repeats(t) == (t, False)
