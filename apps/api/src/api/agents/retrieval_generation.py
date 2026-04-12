import openai

from langsmith import traceable, get_current_run_tree

import re
from typing import List, Dict, Any


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"}
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding


@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_data(query, qdrant_client, k=5):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-item-collection-00",
        query=query_embedding,
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }

@traceable(
    name="filter_retrieved_context",
    run_type="retriever"
)
def filter_context_by_threshold(context, min_score=0.45, max_context_items=3):
    filtered_items = []

    for context_id, chunk, rating, score in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
        context["similarity_scores"],
    ):
        if score >= min_score:
            filtered_items.append(
                {
                    "id": context_id,
                    "chunk": chunk,
                    "rating": rating,
                    "score": score,
                }
            )

    filtered_items = filtered_items[:max_context_items]

    return {
        "retrieved_context_ids": [item["id"] for item in filtered_items],
        "retrieved_context": [item["chunk"] for item in filtered_items],
        "retrieved_context_ratings": [item["rating"] for item in filtered_items],
        "similarity_scores": [item["score"] for item in filtered_items],
        "effective_k": len(filtered_items),
    }

@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)

def process_context(context):
    if not context["retrieved_context"]:
        return "No sufficiently relevant products were found."

    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"-ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_context, question):
    prompt = f"""
You are a precise and helpful shopping assistant.

Your task is to answer the question using ONLY the available products information.

RULES:
- Use only the available products information.
- Do not use external knowledge.
- Do not invent or assume missing details.
- If the answer is not available in the products, say that it is not specified.
- Answer naturally and concisely.

AVAILABLE PRODUCTS:
{preprocessed_context}

QUESTION:
{question}

ANSWER:
"""
    return prompt



@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-5-nano"}
)
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "system", "content": prompt}],
        reasoning_effort="minimal"
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.choices[0].message.content


@traceable(
    name="rag_pipeline",
)
def rag_pipeline(
    question,
    qdrant_client,
    top_k=10,
    min_score=0.45,
    max_context_items=3,
):
    raw_context = retrieve_data(question, qdrant_client, top_k)
    filtered_context = filter_context_by_threshold(
        raw_context,
        min_score=min_score,
        max_context_items=max_context_items,
    )

    preprocessed_context = process_context(filtered_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt).strip()

    final_result = {
        "answer": answer,
        "question": question,
        "raw_retrieved_context_ids": raw_context["retrieved_context_ids"],
        "raw_retrieved_context": raw_context["retrieved_context"],
        "raw_similarity_scores": raw_context["similarity_scores"],
        "retrieved_context_ids": filtered_context["retrieved_context_ids"],
        "retrieved_context": filtered_context["retrieved_context"],
        "similarity_scores": filtered_context["similarity_scores"],
        "effective_k": filtered_context["effective_k"],
        "used_similarity_threshold": min_score,
    }

    return final_result
