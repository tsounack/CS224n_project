import PyPDF2

from transformers import BertTokenizer 

def extract_text_from_pdf(pdf_path, page_number, token, text_len):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        # Access the specific page
        page = pdf_reader.getPage(page_number - 1)  # Adjust page number to 0-based index
        # Extract text from the page
        text = page.extractText()

    # See how many tokens the text is 
    if text_len is not None:
        text = text[:text_len]
    encoded_input = tokenizer(text, return_tensors='pt')
    input_ids = encoded_input['input_ids']
    tokens = input_ids.size(1)

    return text, tokens

# Example usage
pdf_path = "Data/Lakers_Specification.pdf"
page_number = 282
tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
extracted_text, tokens = extract_text_from_pdf(pdf_path, page_number, tokenizer, 2000)
print(extracted_text, tokens)
