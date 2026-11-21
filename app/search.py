"""Search and semantic similarity over the fixed paper corpus."""
from dataclasses import dataclass
import re

import numpy as np
from fastapi import APIRouter, Query

from app.corpus import CORPUS

router = APIRouter(tags=["search"])
STOP_WORDS = {
    "a", "an", "and", "are", "for", "in", "of", "on", "or", "the", "to",
    "ai", "spacecraft", "technique", "techniques", "does", "method", "methods",
    "paper", "question", "this", "use", "used", "what",
}


def stem(word: str) -> str:
    """Reduce the inflections used by the corpus' synthetic search terms."""
    word = re.sub(r"[^a-z0-9]", "", word.lower())
    if word.endswith("tion") and len(word) > 5:
        return word[:-4] + "t"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def key_terms(text: str) -> set[str]:
    return {
        term for term in (stem(token) for token in _tokens(text))
        if term and term not in STOP_WORDS
    }


def analyze(text: str) -> set[str]:
    """Return normalized concepts for search and extractive answering."""
    return key_terms(text)


def _paper_text(paper: dict) -> str:
    derived = paper["derived"]
    fields = [
        paper["title"],
        paper["abstract"],
        *derived["topics"],
        *derived["subtopics"],
        *derived["missions"],
        *derived["spacecraft"],
        *derived["technologies"],
        *derived["methods"],
        *derived["keywords"],
    ]
    return " ".join(str(value) for value in fields if value)


@dataclass(frozen=True)
class SearchEngine:
    vectors: np.ndarray
    vocabulary: dict[str, int]
    idf: np.ndarray

    def rank(self, query: str, limit: int) -> list[tuple[int, float]]:
        query_vector = _vectorize(query, self.vocabulary, self.idf)
        scores = np.clip(self.vectors @ query_vector, 0, 1)
        ranked = np.argsort(-scores)
        return [(int(index), float(scores[index])) for index in ranked[:limit] if scores[index] > 0]

    def search(self, query: str, limit: int | None = None):
        ranked = self.rank(query, len(CORPUS.papers))
        if limit is not None:
            return ranked[:limit]
        query_terms = key_terms(query)
        results = [
            {
                "index": index,
                "relevance": {
                    "overall": round(100 * _overall_score(query_terms, CORPUS.papers[index], score)),
                },
            }
            for index, score in ranked
        ]
        return _interpretation(query), results


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _vectorize(text: str, vocabulary: dict[str, int], idf: np.ndarray) -> np.ndarray:
    vector = np.zeros(len(vocabulary), dtype=float)
    for token in _tokens(text):
        index = vocabulary.get(token)
        if index is not None:
            vector[index] += 1
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector * idf


_texts = [_paper_text(paper) for paper in CORPUS.papers]
_vocabulary = {token: i for i, token in enumerate(sorted(set().union(*map(_tokens, _texts))))}
_document_frequency = np.zeros(len(_vocabulary), dtype=float)
for text in _texts:
    _document_frequency[[ _vocabulary[token] for token in set(_tokens(text)) ]] += 1
_idf = np.log((1 + len(_texts)) / (1 + _document_frequency)) + 1
_matrix = np.vstack([_vectorize(text, _vocabulary, _idf) for text in _texts])
ENGINE = SearchEngine(_matrix, _vocabulary, _idf)


def _overall_score(query_terms: set[str], paper: dict, semantic_score: float) -> float:
    derived = paper["derived"]
    topic_terms = key_terms(" ".join(derived["topics"]))
    entity_terms = key_terms(" ".join(
        derived["missions"] + derived["spacecraft"] + derived["technologies"]
    ))
    paper_terms = key_terms(_paper_text(paper))
    keyword_score = len(query_terms & paper_terms) / max(1, len(query_terms))
    topic_score = len(query_terms & topic_terms) / max(1, len(query_terms))
    entity_score = len(query_terms & entity_terms) / max(1, len(query_terms))
    return 0.35 * semantic_score + 0.25 * keyword_score + 0.30 * topic_score + 0.10 * entity_score


def _interpretation(query: str) -> str:
    query_terms = key_terms(query)
    topics = sorted({
        topic
        for paper in CORPUS.papers
        for topic in paper["derived"]["topics"]
        if query_terms & key_terms(topic)
    })
    return ", ".join(topics) if topics else "No matching concepts"


@router.get("/search")
def search(q: str = "", limit: int = Query(25, ge=1, le=100)):
    query_terms = key_terms(q)
    ranked = []
    for index, semantic_score in ENGINE.rank(q, len(CORPUS.papers)):
        paper = CORPUS.papers[index]
        derived = paper["derived"]
        topic_terms = key_terms(" ".join(derived["topics"]))
        entity_terms = key_terms(" ".join(
            derived["missions"] + derived["spacecraft"] + derived["technologies"]
        ))
        paper_terms = key_terms(_paper_text(paper))
        keyword_score = len(query_terms & paper_terms) / max(1, len(query_terms))
        topic_score = len(query_terms & topic_terms) / max(1, len(query_terms))
        entity_score = len(query_terms & entity_terms) / max(1, len(query_terms))
        overall = (
            0.35 * semantic_score
            + 0.25 * keyword_score
            + 0.30 * topic_score
            + 0.10 * entity_score
        )
        matched = sorted(query_terms & (topic_terms | entity_terms | paper_terms))
        ranked.append({
            "paper": paper,
            "relevance": {
                "overall": round(100 * overall),
                "semantic": round(100 * semantic_score),
                "keyword": round(100 * keyword_score),
                "topic": round(100 * topic_score),
                "entity": round(100 * entity_score),
                "matched_concepts": matched,
            },
        })

    ranked.sort(key=lambda result: -result["relevance"]["overall"])
    ranked = ranked[:limit]
    interpretation_topics = sorted({
        topic
        for paper in CORPUS.papers
        for topic in paper["derived"]["topics"]
        if query_terms & key_terms(topic)
    })
    interpretation = ", ".join(interpretation_topics) if interpretation_topics else "No matching concepts"
    return {"results": ranked, "total": len(ranked), "interpretation": interpretation, "query": q}
