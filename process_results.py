from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Result:
    text: str
    source: Optional[str]
    page: Optional[int]
    distance: Optional[float]


def process_results(results) -> List[Result]:
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    print(metas)
    dists = results["distances"][0]

    processed_results: List[Result] = []
    for text, meta, dist in zip(docs, metas, dists):
        processed_results.append(
            Result(
                text=text,
                source=meta["source"],
                page=meta["page"],
                distance=dist,
            )
        )

    return processed_results
