from fastapi import FastAPI, APIRouter, Depends, Request
import spacy
from spacy.language import Language

router = APIRouter()


def make_app():
    app = FastAPI()
    app.state.NLP = spacy.load("en_core_web_sm")
    app.include_router(router)
    return app


def get_nlp(request: Request) -> Language:
    return request.app.state.NLP


@router.post("/parse")
def parse_texts(
    *, text_batch: list[str], nlp: Language = Depends(get_nlp)
) -> list[dict]:
    with nlp.memory_zone():
        # Put the spaCy call within a separate function, so we can't
        # leak the Doc objects outside the scope of the memory zone.
        output = _process_text(nlp, text_batch)
    return output


def _process_text(nlp: Language, texts: list[str]) -> list[dict]:
    # Call spaCy, and transform the output into our own data
    # structures. This function is called from inside a memory
    # zone, so must not return the spaCy objects.
    docs = list(nlp.pipe(texts))
    return [
        {
            "tokens": [{"text": t.text} for t in doc],
            "entities": [
                {"start": e.start, "end": e.end, "label": e.label_} for e in doc.ents
            ],
        }
        for doc in docs
    ]


app = make_app()