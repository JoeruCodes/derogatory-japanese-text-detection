import argparse
import torch
import pandas as pd
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define the training function
def train(model, dataloader, optimizer, num_epochs):
    model.to('cuda')
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')

    return model

# Define the main function
def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True, type=str, help='Path to the training data file')
    parser.add_argument('--pretrained_model', type=str, default='/model.pth', help='Name or path of the pre-trained BERT model')
    parser.add_argument('--label', type=str, required=True, help='label of your text data')
    parser.add_argument('--text', type=str, required=True, help='your text data')
    parser.add_argument('--output_dir', type=str, default='./', help='Path to the output directory where the fine-tuned model will be saved')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum length of input sequences')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    #parser.add_argument('--gpu', type=int, default=-1, help='Index of the GPU to use. Use -1 for CPU.')
    args = parser.parse_args()

    # Load the training data
    data = pd.read_csv(args.data_file)

    # Instantiate the tokenizer and model
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model_state_dict = torch.load(args.pretrained_model)
    model = model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', state_dict=model_state_dict, num_labels=2)

    # Define the hyperparameters and optimizer
    batch_size = args.batch_size
    max_len = args.max_len
    lr = args.learning_rate
    dataset = CustomDataset(texts=data[args.text], labels=data[args.label], tokenizer=tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the device (GPU or CPU)
    #device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model = train(model, dataloader, optimizer, num_epochs=args.num_epochs)

    # Save the fine-tuned model
    model.save_pretrained(args.output_dir)

if __name__ == '__main__':
   main()

