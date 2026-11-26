import pytest

from khutbah.hadith import HadithDB

H = HadithDB()


@pytest.mark.parametrize(
    "text, collection, ref",
    [
        ("أن تعبد الله كأنك تراه فإن لم تكن تراه فإنه يراك", "nawawi", "2"),
        ("أن تلد الأمة ربتها وأن ترى الحفاة العراة العالة رعاء الشاء يتطاولون في البنيان", "nawawi", "2"),
        ("فإنه جبريل أتاكم يعلمكم دينكم", "nawawi", "2"),
        ("إنما الأعمال بالنيات وإنما لكل امرئ ما نوى", "nawawi", "1"),
        ("فينزل عند المنارة البيضاء شرقي دمشق", "muslim", "2937"),
        # Real Qwen3 output from the Hadith Jibril dars
        ("أنترى الحفاة العرارة العالية تروع الشاء يتطولون في البنين", "nawawi", "2"),
    ],
)
def test_finds_quoted_hadith(text, collection, ref):
    spans = H.find(text)
    assert spans and spans[0].is_hadith(), spans
    h = spans[0].hadith
    if collection:
        assert (h.collection, h.ref) == (collection, ref), h.citation


@pytest.mark.parametrize(
    "text",
    [
        "قال رسول الله صلى الله عليه وسلم",
        "عن عمر بن الخطاب رضي الله عنه قال",
        "أيها الإخوة الكرام اتقوا الله في أنفسكم وأهليكم",
        "اللهم اغفر للمسلمين والمسلمات الأحياء منهم والأموات",
        "وصلى الله على نبينا محمد وعلى آله وصحبه أجمعين",
        "إن الحمد لله نحمده ونستعينه ونستغفره ونعوذ بالله من شرور أنفسنا",
        "وأشهد أن لا إله إلا الله وحده لا شريك له وأشهد أن محمدا عبده ورسوله",
    ],
)
def test_stock_phrases_rejected(text):
    assert not H.find(text), H.find(text)


def test_garbled_closing_dua_is_formula():
    # Real Omnilingual output of the kaffarat al-majlis du'a at the end of a dars
    t = "جبريل عليه السلام أقول خول هذا ونستغفر ركم سبحانكما بحمدك مشهلا لا إله إلا أنت نستغفرك عتوب ريك زكم"
    assert not any(s.is_hadith() for s in H.find(t))


def test_split_quote_accumulates_across_segments():
    from khutbah.pipeline import QuoteFinder

    qf = QuoteFinder(hadith=H)
    # Abu Dawud 4298 quoted in two pieces with English explanation in between (khutbah segs 52, 60)
    _, c1 = qf.process(52, "The Prophet said إن فسطاط المسلمين يوم الملحمة الكبرى", t=300.0)
    _, c2 = qf.process(53, "and he explained what fustat means, a camp", t=310.0)
    _, c3 = qf.process(60, "بمدينة يقال لها دمشق narrated by Abu Dawud", t=340.0)
    assert not c1 and not c2
    assert [c.ref for c in c3] == ["Sunan Abi Dawud 4298"]
    # Repeating it soon after is not a new card
    _, c4 = qf.process(61, "إلى الغوطة إلى جانب مدينة يقال لها دمشق من خير مدائن الشام", t=360.0)
    assert all(not c.new for c in c4)
