from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from utils import extract_text_from_pdf

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-cased-whole-word-masking'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def extract_relevant_pages_embeddings(pdf_path, query, k=2):
    #create a dict of page number to str 
    dictionary = {}

    #preprocess the pdf into a list of documents 
    documents, pages = extract_text_from_pdf(pdf_path)

    # Tokenize input query
    query_tokens = tokenizer.encode_plus(query, add_special_tokens=True, return_tensors="pt")

    # Obtain embeddings for query
    with torch.no_grad():
        query_outputs = model(**query_tokens)
    query_hidden_states = query_outputs.last_hidden_state
    query_pooled_embedding = torch.mean(query_hidden_states, dim=1)

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
        print('Analyzed page ', pages[index])
        
    similarities = []
    for embedding in embedding_list:
        # Compute cosine similarity between text and query embeddings
        similarity_score = cosine_similarity(embedding.numpy(), query_pooled_embedding.numpy())[0][0]
        similarities.append(similarity_score)
    
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
    out_pages, out_docs = extract_relevant_pages_embeddings('Data/Lakers_Specification_TEST.pdf', "What is the steel made out of for framing?", 6)
    print(out_pages, out_docs)






        


    

