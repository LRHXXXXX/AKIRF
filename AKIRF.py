import json
import random
import time
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vllm import LLM, SamplingParams




# ----------------------------
# Agents
# ----------------------------
class KnowledgeAgent:
    """
    Knowledge Agent: extracts key entities from a question and retrieves relevant background
    from a document store and a knowledge graph.

    Corresponds to Stage 1 in the paper: Extraction and Retrieval.
    """

    def __init__(
        self,
        llm_engine: LLM,
        llm_sampling_params: SamplingParams,
        embedding_model: SentenceTransformer,
        df_text: pd.DataFrame,
        df_relation: pd.DataFrame,
    ):
        self.llm = llm_engine
        self.llm_sampling_params = llm_sampling_params
        self.embedding_model = embedding_model

        # Data preparation
        self.df_relation = df_relation
        self.text_list = [row["fragment_content"] for _, row in df_text.iterrows()]

        # Knowledge graph entities
        self.head_entity_list = self.df_relation[" head_entity2id"].tolist()
        self.tail_entity_list = self.df_relation[" tail_entity2id"].tolist()

        # Precompute document embeddings
        print("KnowledgeAgent: Precomputing document embeddings...")
        self.text_embeddings = self.embedding_model.encode(self.text_list)

    def extract_entities(self, question: str, max_retries: int = 3) -> List[str]:
        """
        Extract key entities from the question using a local LLM.
        Output must be JSON: {"key_entities": ["...", "..."]}.
        """
        system_instruction = (
            "You are a domain expert in shipbuilding process knowledge. "
            "Extract the key entities from the user question. "
            "Return ONLY valid JSON in the following schema:\n"
            '{"key_entities": ["entity1", "entity2", "..."]}\n'
            "Do not add any extra text."
        )

        prompt = (
            f"{system_instruction}\n\n"
            f"Question:\n{question}\n"
        )

        try:
            parsed = llm_generate_json(self.llm, prompt, self.llm_sampling_params, max_retries=max_retries)
            entities = parsed.get("key_entities", []) if isinstance(parsed, dict) else []
            return [str(x).strip() for x in entities if str(x).strip()]
        except Exception as e:
            print(f"Entity extraction failed: {e}")
            return []

    def retrieve_triplets(self, entities: List[str], top_k: int = 5) -> List[str]:
        """Retrieve relevant triplets from the knowledge graph."""
        if not entities:
            return []

        retrieved_triplets: List[str] = []

        # Embeddings
        entity_embeddings = self.embedding_model.encode(entities)

        # NOTE: For best performance, precompute these in __init__ in production.
        head_embeddings = self.embedding_model.encode(self.head_entity_list)
        tail_embeddings = self.embedding_model.encode(self.tail_entity_list)

        # Similarities
        head_sims = cosine_similarity(entity_embeddings, head_embeddings)
        tail_sims = cosine_similarity(entity_embeddings, tail_embeddings)

        # Merge and deduplicate
        for i in range(len(entities)):
            h_indices = np.argsort(head_sims[i])[-top_k:][::-1]
            t_indices = np.argsort(tail_sims[i])[-top_k:][::-1]

            for idx in h_indices:
                row = self.df_relation.iloc[idx]
                retrieved_triplets.append(
                    f"[{row[' head_entity2id']}, {row[' relation2id']}, {row[' tail_entity2id']}]"
                )
            for idx in t_indices:
                row = self.df_relation.iloc[idx]
                retrieved_triplets.append(
                    f"[{row[' head_entity2id']}, {row[' relation2id']}, {row[' tail_entity2id']}]"
                )

        return list(set(retrieved_triplets))

    def retrieve_texts(self, question: str, top_k: int = 3, noise_count: int = 0) -> List[str]:
        """Retrieve relevant text fragments from the document store (optionally adding noise)."""
        query_embedding = self.embedding_model.encode([question])
        similarities = cosine_similarity(query_embedding, self.text_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_texts = [self.text_list[idx] for idx in top_indices]

        # Add noise
        if noise_count > 0:
            exclude_indices = set(top_indices)
            candidate_indices = list(set(range(len(self.text_list))) - exclude_indices)

            if len(candidate_indices) >= noise_count:
                noise_indices = random.sample(candidate_indices, noise_count)
                noise_texts = [self.text_list[idx] for idx in noise_indices]
            else:
                noise_texts = [self.text_list[idx] for idx in candidate_indices]

            final_texts = relevant_texts + noise_texts
        else:
            final_texts = relevant_texts

        random.shuffle(final_texts)
        return final_texts

    def get_context(self, question: str, noise_count: int = 0):
        """One-stop retrieval of all background information."""
        entities = self.extract_entities(question)
        triplets = self.retrieve_triplets(entities)
        texts = self.retrieve_texts(question, noise_count=noise_count)

        context_str = (
            "**Retrieved Documents:**\n" + "\n".join(texts) +
            "\n\n**Retrieved KG Triples:**\n" + "\n".join(triplets)
        )
        return context_str, texts, triplets


class ReasonerAgent:
    """
    Reasoner Agent: generates multiple reasoning paths (Multi-path Reasoning).
    Corresponds to Equation (3) in the paper.
    """

    def __init__(self, llm_engine: LLM, sampling_params: SamplingParams):
        self.llm = llm_engine
        self.sampling_params = sampling_params

    def generate_paths(self, prompt: str, num_paths: int = 3) -> List[str]:
        prompts = [prompt] * num_paths
        outputs = self.llm.generate(prompts, self.sampling_params)

        reasoning_paths = []
        for output in outputs:
            generated_text = output.outputs[0].text if output.outputs else ""
            reasoning_paths.append(generated_text)

        return reasoning_paths


class RankingAgent:
    """
    Ranking Agent: reranks retrieved documents/triples using a local LLM (list-wise reranking).
    """

    def __init__(self, llm_engine: LLM, sampling_params: SamplingParams):
        self.llm = llm_engine
        self.sampling_params = sampling_params

    def rerank_content(
        self,
        query: str,
        content_list: List[str],
        content_type: str = "text",
        top_k: int = 3,
        max_retries: int = 3,
    ) -> List[str]:
        if not content_list:
            return []
        if len(content_list) <= top_k:
            return content_list

        items_str = ""
        for i, content in enumerate(content_list):
            items_str += f"[{i}] {content}\n"

        if content_type == "text":
            instruction = (
                "Analyze the relevance between the retrieved document fragments and the user question. "
                "Select the fragments that best help answer the question."
            )
        else:
            instruction = (
                "Analyze the relevance between the retrieved knowledge-graph triples and the user question. "
                "Select the triples most helpful for reasoning."
            )

        prompt = (
            "You are a professional information retrieval reranking assistant.\n\n"
            f"User Question:\n{query}\n\n"
            "Candidate List:\n"
            f"{items_str}\n"
            "Task:\n"
            f"{instruction}\n\n"
            f"Return the indices of the top {top_k} items in strictly descending relevance.\n"
            "Output MUST be a pure JSON array, e.g., [2, 0, 5].\n"
            "Do not explain. Return JSON only."
        )

        # Use a low-temperature configuration for deterministic reranking
        rerank_params = SamplingParams(
            temperature=min(getattr(self.sampling_params, "temperature", 0.2), 0.2),
            top_p=getattr(self.sampling_params, "top_p", 0.9),
            max_tokens=getattr(self.sampling_params, "max_tokens", 256),
        )

        try:
            parsed = llm_generate_json(self.llm, prompt, rerank_params, max_retries=max_retries)

            if isinstance(parsed, dict):
                indices = parsed.get("indices", parsed.get("ids", parsed.get("list", [])))
            elif isinstance(parsed, list):
                indices = parsed
            else:
                indices = []

            reranked_results = []
            for idx in indices:
                try:
                    j = int(idx)
                except Exception:
                    continue
                if 0 <= j < len(content_list):
                    reranked_results.append(content_list[j])

            if not reranked_results:
                return content_list[:top_k]
            return reranked_results[:top_k]

        except Exception as e:
            print(f"Reranking error: {e}. Falling back to the first {top_k} items.")
            return content_list[:top_k]


class EvaluationAgent:
    """
    Evaluation Agent: selects the best answer among multiple reasoning paths.
    Corresponds to Algorithm 3 Line 16: a_final <- Eval({a1...at})
    """

    def __init__(self, llm_engine: LLM, sampling_params: SamplingParams):
        self.llm = llm_engine
        self.sampling_params = sampling_params

    def construct_evaluation_prompt(self, question: str, context_text: str, candidate_answers: List[str]) -> str:
        candidates_str = ""
        for i, ans in enumerate(candidate_answers):
            candidates_str += f"\n[Candidate {i+1}]:\n{ans}\n" + "-" * 30

        prompt = (
            "You are a rigorous expert evaluator in ship design and shipbuilding process knowledge.\n"
            "Your task is to choose the best answer based on the given question and the provided context.\n"
            "The best answer must be accurate, logically consistent, and avoid hallucinations.\n\n"
            "Context:\n"
            f"{context_text}\n\n"
            "Question:\n"
            f"{question}\n"
            f"{candidates_str}\n\n"
            "Evaluation Criteria:\n"
            "1) Factual Accuracy: The answer must be consistent with the provided context.\n"
            "2) Logical Coherence: The reasoning must be clear and valid.\n"
            "3) Completeness: The answer should fully address the question.\n\n"
            "Output Requirements:\n"
            "Return ONLY a JSON object with the following schema:\n"
            '{"best_id": 1, "reason": "concise reason"}\n'
            "best_id is 1-based.\n"
            "Do not output any extra text."
        )
        return prompt

    def select_best_answer(self, question: str, context_str: str, candidate_answers: List[str], max_retries: int = 3) -> str:
        if not candidate_answers:
            return "Generation failed: no candidate answers."
        if len(candidate_answers) == 1:
            return candidate_answers[0]

        eval_prompt = self.construct_evaluation_prompt(question, context_str, candidate_answers)

        # Prefer low temperature for evaluation
        eval_params = SamplingParams(
            temperature=min(getattr(self.sampling_params, "temperature", 0.2), 0.2),
            top_p=getattr(self.sampling_params, "top_p", 0.9),
            max_tokens=getattr(self.sampling_params, "max_tokens", 256),
        )

        try:
            result = llm_generate_json(self.llm, eval_prompt, eval_params, max_retries=max_retries)
            best_id = int(result.get("best_id", 1)) - 1 if isinstance(result, dict) else 0

            if 0 <= best_id < len(candidate_answers):
                return candidate_answers[best_id]
            return candidate_answers[0]

        except Exception as e:
            print(f"Evaluation error: {e}. Falling back to the first answer.")
            return candidate_answers[0]
