import os
import pickle

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# this part of code is duplicated from the work of https://github.com/jlko/semantic_uncertainty

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        
        logits = outputs.logits
        class_scores = F.softmax(logits, dim=1)
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction, class_scores

    def get_entailment(self, text1, text2, *args, **kwargs):
        _, softmax_probs = self.check_implication(text1, text2, *args, **kwargs)
        return softmax_probs[0][2].cpu().detach().numpy().astype('float').item()