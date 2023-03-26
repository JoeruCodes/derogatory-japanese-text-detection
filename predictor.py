import argparse
import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

# Define command-line arguments
parser = argparse.ArgumentParser(description='Detect if a sentence is derogatory or non-derogatory')
parser.add_argument('sentence', type=str, help='Input sentence to classify')

# Load the tokenizer and model
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', num_labels=2)

# Load the saved model state
model.load_state_dict(torch.load('/model.pth'))

# Set the device to CPU or GPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to('cuda')

# Define a function to classify a single sentence
def classify_sentence(sentence):
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=200,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to('cuda')
    attention_mask = encoding['attention_mask'].to('cuda')

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1)

    return pred.item()

# Parse the command-line arguments
args = parser.parse_args()

# Classify the input sentence and print the result
result = classify_sentence(args.sentence)
if result == 0:
    print('The sentence is non-derogatory')
else:
    print('The sentence is derogatory')

