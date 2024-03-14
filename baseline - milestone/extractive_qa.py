import pandas as pd
import torch
from transformers import BertForQuestionAnswering, RobertaForQuestionAnswering, LongformerForQuestionAnswering
from transformers import BertTokenizer, RobertaTokenizer, LongformerTokenizer
from tqdm import tqdm

tqdm.pandas()

class ExtractiveQA:
    def __init__(self, model_name: str, data: pd.DataFrame):
        self.model_name = model_name
        self.model, self.tokenizer = self.load_model()
        # data[['start_gt', 'end_gt']] = data.apply(self._get_tokenized_indexes, axis=1, result_type='expand')
        self.data = data

    def load_model(self):
        if self.model_name == "BERT":
            model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
            tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        elif self.model_name == "RoBERTa":
            model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
            tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
        elif self.model_name == "Longformer":
            model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
            tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        else:
            raise ValueError("Model not supported")
        return model, tokenizer

    def _generate_single(self, row):
        question, paragraph = row["Question"], row["Paragraph"]
        encoding = self.tokenizer(question, paragraph, return_tensors="pt", max_length=512, truncation=True, padding=True)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        if "token_type_ids" in encoding:
            token_type_ids = encoding["token_type_ids"]
        else:
            token_type_ids = None  # For models that don't require token_type_ids
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            start_scores, end_scores = outputs.start_logits, outputs.end_logits

        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        # Ensure end_index is after start_index
        while end_index >= end_scores.size(1) or end_index < start_index or end_index - start_index >= 20:
            end_scores[0][end_index] = float('-inf')
            start_scores[0][start_index] = float('-inf')
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)

        answer = self.tokenizer.decode(input_ids[0][start_index:end_index+1], skip_special_tokens=True)
        return start_index.item(), end_index.item(), answer

    
    def generate(self):
        self.data[['start_pred', 'end_pred', 'model_output']] = self.data.progress_apply(self._generate_single, axis=1, result_type='expand')
        # self.data['Result'] = (self.data['start_pred'].between(self.data['start_gt'], self.data['end_gt']) |
        #                        self.data['end_pred'].between(self.data['start_gt'], self.data['end_gt']))
        self.data['Result'] = self.data.apply(self._verify_outputs, axis=1)
        return self.data
    
    def _get_tokenized_indexes(self, row):
        question = row["Question"]
        paragraph = row["Paragraph"]
        # Use encode_plus to tokenize both question and paragraph
        encoding = self.tokenizer.encode_plus(text=question, text_pair=paragraph, truncation=True, padding="max_length", max_length=512)
        context_tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        answer_tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(row['answer'], add_special_tokens=False)))
        # Find the start and end token indexes
        start_idx = context_tokens.index(answer_tokens[0])
        end_idx = start_idx + len(answer_tokens) - 1
        return start_idx, end_idx
    
    def _verify_outputs(self, row):
        str1, str2 = row['model_output'], row['answer']
        return sum(min(str1.count(char), str2.count(char)) for char in set(str1 + str2)) >= len(str1 + str2) / 3

    def evaluate(self):
        print(f"Accuracy: {self.data['Result'].mean()}")