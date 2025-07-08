import spacy


def speech_tagging(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for entity in doc.ents:
        print(entity.text, entity.label_)
