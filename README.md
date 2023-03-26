# Japanese Derogatory Sentence Detector

This project contains a neural network model trained to detect if a given Japanese sentence is derogatory or non-derogatory. The model was trained using a dataset of Japanese sentences labeled as either derogatory or non-derogatory.

**PS: THIS MODEL SUCKS AT DETECTING MISOGYNY (WILL BE FIXED IN V2 OF THIS MODEL)**


## Requirements

To use the Japanese Derogatory Sentence Detector, you will need:

    1. Python 3.x
    2. PyTorch
    3. Transformers
    4. Pandas
    5. Fugashi
    6. mecab
    7. ipadic dictionary
    
You can install the required packages by running:

     pip install -r requirements.txt.
    
**Didn't use ipadic NEologd because it was causing learning issues, and mainly because my training data had emoticons as well... Apparently, using the BertTokenizer or BertJapaneseTokenizer with the default mecab ipadic dictionary does the job as well with better accuracy and performance...**


## Usage

To use the Japanese Derogatory Sentence Detector, you can run the predictor.py script with a Japanese sentence as a command-line argument.


>Here's an example usage:

    python predictor.py "今日の天気はどうですか？"

>Output:

    The sentence is non-derogatory
    
>**NOTE: The model in reality predicts 0 if the given sentence is non-derogatory and 1 if its derogatory**



## Training

### Running train.py with Command Line Arguments
This script trains a BERT model for sequence classification using a Japanese tokenizer. It takes a CSV file containing text data and their corresponding labels, and outputs a fine-tuned model that can be used for inference. The script requires the following command line arguments:

+ --data_file (required): Path to the CSV file containing the training data.
+ --label (required): The name of the column in the CSV file that contains the labels.
+ --text (required): The name of the column in the CSV file that contains the text data.
+ --pretrained_model (default: "/model.pth"): The path to the pre-trained model for tuning.
+ --output_dir (default: "./"): The path to the output directory where the fine-tuned model will be saved.
+ --batch_size (default: 16): The batch size for training.
+ --max_len (default: 200): The maximum length of input sequences.
+ --learning_rate (default: 2e-5): The learning rate for the optimizer.
+ --num_epochs (default: 200): The number of epochs for training.

#### Example:

     python train.py \
     --data_file data/train_data.csv \
     --label label \
     --text text \
     --pretrained_model pretrained/bert-base-japanese-whole-word-masking.pth \
     --output_dir models \
     --batch_size 32 \
     --max_len 256 \
     --learning_rate 3e-5 \
     --num_epochs 5 \

## Model

The model architecture used for the Japanese Derogatory Sentence Detector is the BERT (Bidirectional Encoder Representations from Transformers) model, which is a pre-trained transformer-based neural network model developed by Google. The BERT model was fine-tuned using the cl-tohoku/bert-base-japanese-whole-word-masking pre-trained model. The model was trained using PyTorch and achieved an accuracy of 99.4% on the test set. It wasn't trained on misogynistic comments, so it doesn't detect it well... It will be fixed in the V2 of this model...


## Acknowledgements

1. This project is based on the Transformers library by Hugging Face: https://github.com/huggingface/transformers
2. The BERT model used for this project is based on the pre-trained model by the Tohoku University NLP Lab: https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as you include the original license file.
