from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from utils1 import extract_text_from_pdf

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-cased-whole-word-masking'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def create_embeddings(pdf_path: str):
    """
    Takes a pdf and converts each page into embeddings

    Args:
    pdf_path (str): a string representing the filepath of the embedding 

    Returns:
    embedding_list (List[List[float]]): a list of embeddings for each subdivided
    embedding_idx (List[str]): a list of corresponding pages for each embedding
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
    
    return embedding_list, embedding_idx

def extract_relevant_pages_embeddings(pdf_path, query, embedding_list, k=2):
    #preprocess the pdf into a list of documents 
    documents, pages = extract_text_from_pdf(pdf_path)

    # Tokenize input query
    query_tokens = tokenizer.encode_plus(query, add_special_tokens=True, return_tensors="pt")

    # Obtain embeddings for query
    with torch.no_grad():
        query_outputs = model(**query_tokens)
    query_hidden_states = query_outputs.last_hidden_state
    query_pooled_embedding = torch.mean(query_hidden_states, dim=1)

        
    similarities = []
    for index, embedding in enumerate(embedding_list):
        # Compute cosine similarity between text and query embeddings
        similarity_score = cosine_similarity(embedding.numpy(), query_pooled_embedding.numpy())[0][0]
        similarities.append(similarity_score)
        print('Analyzed page ', pages[index])
    
    #zip both list of documents and similarities togtether
    combined = list(zip(similarities, pages, documents))

    # Sort based on the values in list 'a'
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    # Get the first k tuples from results
    first_k_results = sorted_combined[:k]

    # Unzip the tuples
    out_scores, out_pages, out_docs = zip(*first_k_results)

    return list(out_docs), list(out_pages)

if __name__ == "__main__":
    embeddings, embedding_idx = create_embeddings('Data/Lakers_Specification.pdf')
    out_docs, out_pages = extract_relevant_pages_embeddings('Data/Lakers_Specification.pdf', "what is the address of the architect?", embeddings, 5)
    print(out_pages,'\n', out_docs)






        


    

