from rank_bm25 import BM25Okapi
import PyPDF2

def preprocess(text):
    # Tokenize the text
    tokens = text.lower().split()
    # You can add additional preprocessing steps like removing stop words, stemming, lemmatization, etc.
    return tokens

def extract_text_from_pdf(pdf_path):
    text_list = []
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
            # Append the text to the list
            text_list.append(text)
    return text_list

def bm25_obj(pdf_path):
    #preprocess the pdf into a list of documents 
    documents = extract_text_from_pdf(pdf_path)

    # Preprocess the documents
    tokenized_docs = [preprocess(doc) for doc in documents]

    # Create BM25Okapi object
    bm25 = BM25Okapi(tokenized_docs)

    return bm25



def extract_relevant_pages_bm25(bm25, pdf_path, query, k=2): 
    #create a dict of page number to str 
    dictionary = {}

    #preprocess the pdf into a list of documents 
    documents = extract_text_from_pdf(pdf_path)

    # Preprocess the query
    query_tokens = preprocess(query)

    # Get the BM25 scores for the query
    scores = bm25.get_scores(query_tokens)

    # Combine the scores with document indices
    results = list(zip(range(len(documents)), scores))

    # Sort the results by BM25 score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Print the top k results
    for idx, score in results[:k]:
        # print(f"Document {idx + 1}: {documents[idx]} - BM25 Score: {score}")
        doc_dict = {}
        doc_dict['score'] = score
        doc_dict['text'] = documents[idx]
        dictionary[idx + 1] = doc_dict
    return dictionary

if __name__ == "__main__":
    bm25 = bm25_obj('Data/Lakers_Specification.pdf')
    dictionary = extract_relevant_pages_bm25(bm25, 'Data/Lakers_Specification.pdf', "what is the address of the architect?", 2)
    print(dictionary)

