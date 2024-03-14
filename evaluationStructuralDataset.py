from bm25 import extract_relevant_pages_bm25, bm25_obj
import pandas as pd
from bm25_mugi import MuGI
from cosine_similarity import extract_relevant_pages_embeddings, create_embeddings
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from utils1 import extract_text_from_pdf
from score import score

def bm25_evaluation(csv_path, pdf_path, k, model, tokenizer):
    #read the csv
    df = pd.read_csv(csv_path)
    #read the pdf
    docs, doc_idx = extract_text_from_pdf(pdf_path)
    # get bm25 obj
    bm25 = bm25_obj(pdf_path)
    # mugi_embed = MuGI('dummy')
    # dict_embeddings = mugi_embed.create_embeddings(pdf_path, tokenizer, model)
    # embedding_list = list(dict_embeddings.values())
    # Save embeddings to cache
    # with open('embeddings_cache.pkl', 'wb') as f:
    #     pickle.dump(embedding_list, f)

    with open('embeddings_cache.pkl', 'rb') as f:
        embedding_list = pickle.load(f)

    dict_embeddings = dict(zip(docs, embedding_list))

    pages_bm25 = []
    pages_bm25_mugi = []
    pages_cosine_similarity = []
    pages_bm25_mugi_no_rerank = []
    #iterate through the rows of the df
    for index, row in df.iterrows():
        #grab the query 
        query = row['Question']

        # #use bm25 to grab the relevant page 
        # relevant_pages, page_idx = extract_relevant_pages_bm25(bm25, pdf_path, query, k)
        # pages_bm25.append(page_idx)

        #use cosine similarity to grab the relevant pages 
        # out_docs, out_pages = extract_relevant_pages_embeddings(pdf_path, query, embedding_list, k)
        # pages_cosine_similarity.append(out_pages)

        #use bm25 with mugi to grab the relevant pages
        mugi = MuGI(query)
        # sorted_paragraphs, paragraph_idx = mugi._run_bm25(pdf_path=pdf_path, model_name='gpt-3.5-turbo-1106', num_queries=1, num_repetitions=3, dict_embeddings = dict_embeddings, model=model, tokenizer=tokenizer, bm25=bm25, k=k)
        s_p_norerank, p_i_norerank = mugi._run_bm25_no_rerank(pdf_path=pdf_path, model_name='gpt-3.5-turbo-1106', num_queries=5, num_repetitions=5, bm25=bm25, k=k)
        # pages_bm25_mugi.append(paragraph_idx)
        pages_bm25_mugi_no_rerank.append(p_i_norerank)
        print('Processed query ', index+1, 'out of ', len(df))
    #create a new results column 
    # df['bm25_k=3_pages'] = pages_bm25
    # df['bm25_mugi_k=3_pages_num_queries=1'] = pages_bm25_mugi
    df['bm25_mugi_k=3_pages_num_queries=5_norerank'] = pages_bm25_mugi_no_rerank
    # df['cosine_similarity_k=3'] = pages_cosine_similarity
    return df 


if __name__ == "__main__":
    #inputs 
    csv_path = 'Data/Lakers_QA_Dataset - BM25_Evaluation_6.csv'#'Data/Lakers_QA_Dataset - Sheet1.csv'
    pdf_path = 'Data/Lakers_Specification.pdf'
    k=3
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-large-cased-whole-word-masking'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    #call bm25 function 
    # df = bm25_evaluation(csv_path, pdf_path, k, model, tokenizer)

    csv_path = 'Data/Lakers_QA_Dataset - BM25_Evaluation_7.csv'  # Specify the CSV file path
    # df.to_csv(csv_path, index=False)  # Save the DataFrame to a CSV file
    
    df_rankings = score(csv_path = csv_path)
    score_dict = df_rankings.compute_precision()

    print(score_dict)
