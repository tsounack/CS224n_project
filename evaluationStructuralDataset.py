from bm25 import extract_relevant_pages_bm25, bm25_obj
import pandas as pd


def bm25_evaluation(csv_path, pdf_path, k):
    #read the csv
    df = pd.read_csv(csv_path)

    #get bm25 obj
    bm25 = bm25_obj(pdf_path)

    pages = []
    #iterate through the rows of the df
    for index, row in df.iterrows():
        #grab the query 
        query = row['Question']

        #use bm25 to grab the relevant page 
        dict_pages = extract_relevant_pages_bm25(bm25, pdf_path, query, k)
        
        #access the pages from the dictionar
        page_range = list(dict_pages.keys())
        pages.append(page_range)

        print('Processed query ', index+1, 'out of ', len(df))
    #create a new results column 
    df['bm25_k=3_pages'] = pages
    return df 


if __name__ == "__main__":
    #inputs 
    csv_path = 'Data/Lakers_QA_Dataset - Sheet1.csv'
    pdf_path = 'Data/Lakers_Specification.pdf'
    k=3

    #call bm25 function 
    df = bm25_evaluation(csv_path, pdf_path, k)

    csv_path = 'Data/Lakers_QA_Dataset - BM25_Evaluation_2.csv'  # Specify the CSV file path
    df.to_csv(csv_path, index=False)  # Save the DataFrame to a CSV file

