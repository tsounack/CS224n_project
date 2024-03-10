import os
import torch
from openai import OpenAI
from mugi.mugi import generate_pseudo_references
from rank_bm25 import BM25Okapi
from utils1 import extract_text_from_pdf, preprocess
from typing import List
from transformers import BertTokenizer, BertModel


class MuGI:
    """
    MuGI (Multi-Query Generation and Integration) is a class that represents a query generator and ranker.
    It takes a query as input and generates multiple pseudo-references based on a given model.
    The generated queries are then used to rank entries in a PDF document using the BM25 algorithm.

    Methods:
        generate_total_query: Generates the total query by combining the initial query and the generated queries.
        rank_entries: Ranks entries in a PDF document based on the total query.
        rerank_entries: Reranks the entries in the given page_to_results dictionary based on their similarity to the total_query.
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
            out_docs (List[str]): reranked sorted list of strings representing text
            out_pages (List[str]): corresponding pages to the text in out_docs
        """
        page_to_results = {}
        entries, entry_idx = extract_text_from_pdf(pdf_path)
        query_tokens = preprocess(total_query)
        bm25 = self._generate_bm25_obj(pdf_path)
        ranking = bm25.get_scores(query_tokens)
        results = list(zip(ranking, entry_idx, entries))
        results.sort(key=lambda x: x[0], reverse=True)
        first_k_results = results[:k]
        out_scores, out_pages, out_docs = zip(*first_k_results)
        return list(out_docs), list(out_pages)
    
    def rerank_entries(self, total_query: str, entries: List[str], entry_idx: List[str], model: object, tokenizer: object):
            """
            Reranks the entries in the given page_to_results dictionary based on their similarity to the total_query.
            
            Args:
                total_query (str): The total query string.
                entries (List[str]): A list of strings representing the top K text 
                entry_idx (List[str]): A list of pages representing the respective pages corresponding to entries
                model (object): The model used for embedding the queries and entries.
                tokenizer (object): The tokenizer used for tokenizing the queries and entries.
            
            Returns:
                out_docs (List[str]): reranked sorted list of strings representing text
                out_pages (List[str]): corresponding pages to the text in out_docs
            """
            tokenized_query = tokenizer.encode_plus(total_query, add_special_tokens=True, return_tensors="pt")
            with torch.no_grad():
                embedded_query = model(**tokenized_query)
            query_hidden_states = embedded_query.last_hidden_state
            query_pooled_embedding = torch.mean(query_hidden_states, dim=1)
            embedding_entries = []
            for entry in entries:
                tokenized_entry = tokenizer.encode_plus(entry, add_special_tokens=True, return_tensors="pt")
                with torch.no_grad():
                    embedded_entry = model(**tokenized_entry)
                entry_hidden_states = embedded_entry.last_hidden_state
                entry_pooled_embedding = torch.mean(entry_hidden_states, dim=1)
                embedding_entries.append(entry_pooled_embedding)
            similarity_scores = []
            for embedding in embedding_entries:
                similarity_score = torch.nn.functional.cosine_similarity(embedding, query_pooled_embedding)
                similarity_scores.append(similarity_score.item())
            sorted_ranked = sorted(zip(similarity_scores, entry_idx, entries), key=lambda x: x[0], reverse=True)
            first_k_results = sorted_ranked
            out_scores, out_pages, out_docs = zip(*first_k_results)
            return list(out_docs), list(out_pages)

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
    
    def _run_bm25(self, pdf_path: str, model_name: str, num_queries: int, num_repetitions: int, model: object, tokenizer: object, k: int=2,):
        """
        Runs the bm25 with mugi algorithim end to end assembling the different methods 

        Args:
            pdf_path (str): The path to the PDF document.
            model_name (str): The model to use: 'gpt-3.5-turbo-1106'
            num_queries (int): the number of additional queries 
            num_repitions (int): the number of times to repeat the query
            model (object): The model used for embedding the queries and entries.
            tokenizer (object): The tokenizer used for tokenizing the queries and entries. 
            k (int, optional): The number of top-ranked entries to return. Defaults to 2.

        Returns:
            out_docs (List[str]): sorted list of strings representing text based on relevance to query 
            out_pages (List[str]): corresponding pages to the text in out_docs
        """
        total_query = self.generate_total_query(model_name, num_queries, num_queries)
        print(total_query)
        bm25_sorted, bm25_pages = self.rank_entries(pdf_path, total_query, k)
        reranked_sorted, pages = self.rerank_entries(total_query, bm25_sorted, bm25_pages, model, tokenizer)
        return reranked_sorted, pages

if __name__ == "__main__":
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-large-cased-whole-word-masking'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    mugi = MuGI("what is the address of the architect?")
    sorted_paragraphs, paragraph_idx = mugi._run_bm25('Data/Lakers_Specification.pdf', 'gpt-3.5-turbo-1106', 1, 3, model, tokenizer, 5)
    print(sorted_paragraphs, '\n', paragraph_idx)