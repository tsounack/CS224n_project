import PyPDF2
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
model_name = 'bert-large-cased-whole-word-masking'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def extract_text_from_pdf(pdf_path):
    '''Takes a pdf file path and splits it into a list of strings where each string is less than 512 tokens
    input: pdf file path
    
    output: 
    text_list: list of strings representing text from a page or half page
    page_num_list: list of corresponding page numbers which match to text_list 1 to 1

    '''
    text_list = []
    page_num_list = []
    # Open the PDF
    with open(pdf_path, 'rb') as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        # Iterate through each page
        for page_num in range(pdf_reader.numPages):
            # Get the page
            page = pdf_reader.getPage(page_num)
            # Extract text from the page
            text = page.extractText()

            #tokenize the text on the page
            text_tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt")
            input_ids = text_tokens['input_ids']
            tokens = input_ids.size(1)

            if tokens >= 512:
                text_list.append(text[:int(len(text)/2)])
                page_num_list.append(page_num + 1)

                text_list.append(text[int(len(text)/4):int(3*len(text)/4)])
                page_num_list.append(page_num + 1)

                text_list.append(text[int(len(text)/2):])
                page_num_list.append(page_num + 1)
            else: 
                text_list.append(text)
                page_num_list.append(page_num + 1) 

    return text_list, page_num_list

def preprocess(text):
    """Takes a string and tokenizes them based on white spaces
    
    inputs:
    text: a string
    
    outputs: 
    tokens: list of strings"""
    # Tokenize the text
    tokens = text.lower().split()
    # You can add additional preprocessing steps like removing stop words, stemming, lemmatization, etc.
    return tokens


# def rerank(dictionary, query):
#     #create a list of pages and text
#     pages = []
#     documents = []

#     for key, value in dictionary.items():
#         pages.append(key)
#         documents.append(value['text'])

#     # Tokenize input query
#     query_tokens = tokenizer.encode_plus(query, add_special_tokens=True, return_tensors="pt")

#     # Obtain embeddings for query
#     with torch.no_grad():
#         query_outputs = model(**query_tokens)
#     query_hidden_states = query_outputs.last_hidden_state
#     query_pooled_embedding = torch.mean(query_hidden_states, dim=1)

#     #iterate through the documents 
#     embedding_list = []
#     for index, doc in enumerate(documents):
#         text_tokens = tokenizer.encode_plus(doc, add_special_tokens=True, return_tensors="pt")
#         input_ids = text_tokens['input_ids']

#         tokens = input_ids.size(1)
#         # Obtain embeddings for text
#         with torch.no_grad():
#             text_outputs = model(**text_tokens)
#         text_hidden_states = text_outputs.last_hidden_state
#         text_pooled_embedding = torch.mean(text_hidden_states, dim=1)
#         embedding_list.append(text_pooled_embedding)
#         print('Analyzed page ', pages[index])
        
#     similarities = []
#     for embedding in embedding_list:
#         # Compute cosine similarity between text and query embeddings
#         similarity_score = cosine_similarity(embedding.numpy(), query_pooled_embedding.numpy())[0][0]
#         similarities.append(similarity_score)
    
#     #zip both list of documents and similarities togtether
#     combined = list(zip(similarities, pages, documents))

#     # Sort based on the values in list 'a'
#     sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

#     for sim, page, doc in sorted_combined:
#         doc_dict={}
#         doc_dict['score'] = str(sim)
#         doc_dict['text'] = doc
#         dictionary[page] = doc_dict

#     return dictionary

# if __name__ == "__main__":
#     import json
#     # Open the file
#     with open('embeddings.json', 'r') as f:
#         # Load the JSON data from the file into a dictionary
#         data = json.load(f)

#     dic = rerank(data, "What is the steel made out of for framing?")
#     print(dic)
