import os
from openai import OpenAI
from mugi.mugi import generate_pseudo_references
from rank_bm25 import BM25Okapi
from utils import extract_text_from_pdf, preprocess

class MuGI:
    """
    MuGI (Multi-Query Generation and Integration) is a class that represents a query generator and ranker.
    It takes a query as input and generates multiple pseudo-references based on a given model.
    The generated queries are then used to rank entries in a PDF document using the BM25 algorithm.

    Methods:
        generate_total_query: Generates the total query by combining the initial query and the generated queries.
        rank_entries: Ranks entries in a PDF document based on the total query.
    """

    def __init__(self, query: str):
        """
        Initializes a new instance of the MuGI class.

        Args:
            query (str): The initial query string.
        """
        self.query = {
            1: {'title': query}
        }
    
    def generate_total_query(self, model_name: str, num_queries: int, num_repetitions: int):
        """
        Generates the total query by combining the initial query and the generated queries.

        Args:
            model_name (str): The name of the model used for query generation.
            num_queries (int): The number of pseudo-references to generate.
            num_repetitions (int): The number of times to repeat the initial query.

        Returns:
            str: The total query string.
        """
        new_queries = generate_pseudo_references(model_name, self.query, num_queries)
        queries, gen_key = new_queries
        key = list(queries.keys())[0] # assumes only one query
        initial_query = queries[key]['title']
        generated_queries = ' '.join(queries[key]['gen_cand_gpt35'])
        num_repetitions = (len(generated_queries)//len(initial_query))//num_repetitions
        total_query = (initial_query + ' ') * num_repetitions + generated_queries
        return total_query

    def rank_entries(self, pdf_path: str, total_query: str, k: int = 2):
        """
        Ranks entries in a PDF document based on the total query.

        Args:
            pdf_path (str): The path to the PDF document.
            total_query (str): The total query string.
            k (int, optional): The number of top-ranked entries to return. Defaults to 2.

        Returns:
            dict: A dictionary mapping page numbers to the corresponding ranked results.
        """
        page_to_results = {}
        entries, entry_idx = extract_text_from_pdf(pdf_path)
        query_tokens = preprocess(total_query)
        bm25 = self._generate_bm25_obj(pdf_path)
        ranking = bm25.get_scores(query_tokens)
        results = list(zip(entry_idx, ranking))
        results.sort(key=lambda x: x[1], reverse=True)
        for idx, score in results[:k]:
            page_to_results[idx] = {
                'score': score,
                'text': entries[idx]
            }
        return page_to_results

    def _generate_bm25_obj(self, pdf_path: str):
        """
        Generates a BM25 object for ranking entries in a PDF document.

        Args:
            pdf_path (str): The path to the PDF document.

        Returns:
            BM25Okapi: The BM25 object.
        """
        documents, _ = extract_text_from_pdf(pdf_path)
        tokenized_docs = [preprocess(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        return bm25