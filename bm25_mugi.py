import os
import torch
from openai import OpenAI
from mugi.mugi import generate_pseudo_references
from rank_bm25 import BM25Okapi
from utils1 import extract_text_from_pdf, preprocess
from typing import List, Dict
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
            str: The initial query string. 
        """
        new_queries = generate_pseudo_references(model_name, self.query, num_queries)
        queries, gen_key = new_queries
        key = list(queries.keys())[0] # assumes only one query
        initial_query = queries[key]['title']
        generated_queries = ' '.join(queries[key]['gen_cand_gpt35'])
        num_repetitions = (len(generated_queries)//len(initial_query))//num_repetitions
        total_query = (initial_query + ' ') * num_repetitions + generated_queries
        return total_query, initial_query

    def rank_entries(self, pdf_path: str, total_query: str, bm25: object, k: int = 2,):
        """
        Ranks entries in a PDF document based on the total query.

        Args:
            pdf_path (str): The path to the PDF document.
            total_query (str): The total query string.
            bm25 (object): a bm25 ovbject
            k (int, optional): The number of top-ranked entries to return. Defaults to 2.

       Returns:
            out_docs (List[str]): reranked sorted list of strings representing text
            out_pages (List[str]): corresponding pages to the text in out_docs
        """
        page_to_results = {}
        entries, entry_idx = extract_text_from_pdf(pdf_path)
        query_tokens = preprocess(total_query)
        ranking = bm25.get_scores(query_tokens)
        results = list(zip(ranking, entry_idx, entries))
        results.sort(key=lambda x: x[0], reverse=True)
        first_k_results = results[:k]
        out_scores, out_pages, out_docs = zip(*first_k_results)
        return list(out_docs), list(out_pages)

    def create_embeddings(self, pdf_path: str, tokenizer: object, model: object):
        """
        Takes a pdf and converts each page into embeddings

        Args:
        pdf_path (str): a string representing the filepath of the embedding
        tokenizer (object): The tokenizer used for tokenizing the queries and entries. 
        model (object): The model used for embedding the queries and entries.

        Returns:
        dict_embeddings (Dictionary[str:List[int]]): a dictionary of strings to embeddings 
        """
        #preprocess the pdf into a list of documents 
        documents, embedding_idx = extract_text_from_pdf(pdf_path)

        #iterate through the documents 
        embedding_list = []
        for index, doc in enumerate(documents):
            text_tokens = tokenizer.encode_plus(doc, add_special_tokens=True, return_tensors="pt")
            input_ids = text_tokens['input_ids']
            tokens = input_ids.size(1)
            if tokens >= 512:
                #break the page in half and then feed both halfs into mean 
                text_tokens1 = tokenizer.encode_plus(doc[:int(len(doc)/2)], add_special_tokens=True, return_tensors="pt")
                text_tokens2 = tokenizer.encode_plus(doc[int(len(doc)/2):], add_special_tokens=True, return_tensors="pt")
                # Obtain embeddings for text
                with torch.no_grad():
                    text_outputs1 = model(**text_tokens1)
                    text_outputs2 = model(**text_tokens2)
                text_hidden_states1 = text_outputs1.last_hidden_state
                text_hidden_states2 = text_outputs2.last_hidden_state
                combined_hidden_states = torch.cat([text_hidden_states1, text_hidden_states2], dim=1)
                text_pooled_embedding = torch.mean(combined_hidden_states, dim=1)
            else: 
                # Obtain embeddings for text
                with torch.no_grad():
                    text_outputs = model(**text_tokens)
                text_hidden_states = text_outputs.last_hidden_state
                text_pooled_embedding = torch.mean(text_hidden_states, dim=1)
            embedding_list.append(text_pooled_embedding)
            print('Creating embedding for ', embedding_idx[index])

            #create a dictionary of documents and embedding_list 
            zipped = zip(documents, embedding_list)
            dict_embeddings = dict(zipped)
        return dict_embeddings
    

    def rerank_entries(self, total_query: str, entries: List[str], entry_idx: List[str], dict_embeddings: Dict[str, List[int]], model: object, tokenizer: object, k: int=3):
            """
            Reranks the entries in the given page_to_results dictionary based on their similarity to the total_query.
            
            Args:
                total_query (str): The total query string.
                entries (List[str]): A list of strings representing the top K text 
                entry_idx (List[str]): A list of pages representing the respective pages corresponding to entries
                dict_embeddings: (Dictionary[str:List[int]]): a dictionary of strings to embeddings 
                model (object): The model used for embedding the queries and entries.
                tokenizer (object): The tokenizer used for tokenizing the queries and entries.
                k (int, optional): The number of top-ranked entries to return. Defaults to 2.
            
            Returns:
                out_docs (List[str]): reranked sorted list of strings representing text
                out_pages (List[str]): corresponding pages to the text in out_docs
            """
            tokenized_query = tokenizer.encode_plus(total_query, add_special_tokens=True, return_tensors="pt")
            with torch.no_grad():
                embedded_query = model(**tokenized_query)
            query_hidden_states = embedded_query.last_hidden_state
            query_pooled_embedding = torch.mean(query_hidden_states, dim=1)
            embedding_entries = [dict_embeddings[entry] for entry in entries]
            similarity_scores = []
            for embedding in embedding_entries:
                similarity_score = torch.nn.functional.cosine_similarity(embedding, query_pooled_embedding)
                similarity_scores.append(similarity_score.item())
            sorted_ranked = sorted(zip(similarity_scores, entry_idx, entries), key=lambda x: x[0], reverse=True)
            first_k_results = sorted_ranked[:k]
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
    
    def _run_bm25_rerank(self, pdf_path: str, model_name: str, num_queries: int, num_repetitions: int, dict_embeddings: Dict[str, List[int]], model: object, tokenizer: object, bm25: object, k: int=2,):
        """
        Runs the bm25 with mugi algorithim end to end assembling the different methods 

        Args:
            pdf_path (str): The path to the PDF document.
            model_name (str): The model to use: 'gpt-3.5-turbo-1106'
            num_queries (int): the number of additional queries 
            num_repitions (int): the number of times to repeat the query
            dict_embeddings: (Dictionary[str:List[int]]): a dictionary of strings to embeddings
            model (object): The model used for embedding the queries and entries.
            tokenizer (object): The tokenizer used for tokenizing the queries and entries. 
            bm25 (object): a bm25 object
            k (int, optional): The number of top-ranked entries to return. Defaults to 2.

        Returns:
            out_docs (List[str]): sorted list of strings representing text based on relevance to query 
            out_pages (List[str]): corresponding pages to the text in out_docs
        """
        total_query, initial_query = self.generate_total_query(model_name, num_queries, num_queries)
        print(total_query)
        bm25_sorted, bm25_pages = self.rank_entries(pdf_path, total_query, bm25, 100)
        print(bm25_pages)
        reranked_sorted, pages = self.rerank_entries(total_query, bm25_sorted, bm25_pages, dict_embeddings, model, tokenizer, k)
        return reranked_sorted, pages
    
    def _run_bm25_no_rerank(self, pdf_path: str, model_name: str, num_queries: int, num_repetitions: int, bm25: object, k: int=2,):
        """
        Runs the bm25 with mugi algorithim end to end assembling the different methods 

        Args:
            pdf_path (str): The path to the PDF document.
            model_name (str): The model to use: 'gpt-3.5-turbo-1106'
            num_queries (int): the number of additional queries 
            num_repitions (int): the number of times to repeat the query
            bm25 (object): a bm25 object
            k (int, optional): The number of top-ranked entries to return. Defaults to 2.

        Returns:
            out_docs (List[str]): sorted list of strings representing text based on relevance to query 
            out_pages (List[str]): corresponding pages to the text in out_docs
        """
        total_query, initial_query = self.generate_total_query(model_name, num_queries, num_queries)
        print(total_query)
        bm25_sorted, bm25_pages = self.rank_entries(pdf_path, total_query, bm25, k)
        return bm25_sorted, bm25_pages

if __name__ == "__main__":
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-large-cased-whole-word-masking'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    mugi = MuGI("what is the address of the architect?")
    dict_embeddings = mugi.create_embeddings('Data/Lakers_Specification.pdf', tokenizer, model)
    bm25 = mugi._generate_bm25_obj('Data/Lakers_Specification.pdf')
    sorted_paragraphs, paragraph_idx = mugi._run_bm25_rerank('Data/Lakers_Specification.pdf', 'gpt-3.5-turbo-1106', 1, 3, dict_embeddings, model, tokenizer,bm25, 3)
    # print(sorted_paragraphs, '\n', paragraph_idx)
    print(paragraph_idx)