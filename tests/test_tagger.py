from src.tagging.tags import Tagger


def test_tagger_assigns_emotion_and_color():
    tagger = Tagger()
    tags = tagger.tag_segment("I am happy today")

    assert tags["emotion"] == "joy"
    assert tags["color"] == "gold"
