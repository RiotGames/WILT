import numpy as np
import openai
from functools import lru_cache
from retry import retry

@lru_cache(maxsize=10000)
@retry(tries=3)
def embed(text):
    client = openai.OpenAI()

    response = client.embeddings.create(
        model="text-embedding-3-large", input=[text])
    return response.data[0].embedding

# Reference: https://github.com/aidanmclaughlin/Aidan-Bench/blob/main/main.py#L82
def get_novelty_scores(conversation):
    model_responses = [reply["content"] for reply in conversation 
                             if reply["role"] == "assistant"]

    novelties = []
    seen = []
    embs = []

    for reply in model_responses:
        if not seen:
            novelties.append(1.0)
            seen.append(reply)
            embs.append(embed(reply))
            continue
        
        emb = embed(reply)

        similarities = [
            float(np.dot(emb, old_emb) /
            (np.linalg.norm(emb) * np.linalg.norm(old_emb)))
            for old_emb in embs
        ]

        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity

        embs.append(emb)
        novelties.append(novelty)
        seen.append(reply)

    return novelties
