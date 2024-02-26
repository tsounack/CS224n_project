import fitz

from transformers import BertTokenizer 


def extract_text_from_pdf(pdf_path, page_number, token, text_len):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Access the specific page
    page = doc.load_page(page_number - 1)  # Adjust page number to 0-based index

    # Extract text from the page
    text = page.get_text()

    #see how many tokens the text is 
    if text_len != None:
        text = text[:text_len] 
    encoded_input = tokenizer(text, return_tensors='pt')
    input_ids = encoded_input['input_ids']
    tokens = input_ids.size(1)
    # Close the document
    doc.close()

    return text, tokens

# Example usage
pdf_path = "Data/Lakers_Specification.pdf"
page_number = 30
tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
extracted_text, tokens = extract_text_from_pdf(pdf_path, page_number, tokenizer, 2000)
print(extracted_text, tokens)

