# Japanese Derogatory Sentence Detector

This project contains a neural network model trained to detect if a given Japanese sentence is derogatory or non-derogatory. The model was trained using a dataset of Japanese sentences labeled as either derogatory or non-derogatory.

**PS: THIS MODEL SUCKS AT DETECTING MISOGYNY (WILL BE FIXED IN V2 OF THIS MODEL)**


## Requirements

To use the Japanese Derogatory Sentence Detector, you will need:

    1. Python 3.x
    2. PyTorch
    3. Transformers
    4. Pandas

You can install the required packages by running pip install -r requirements.txt.



## Usage

To use the Japanese Derogatory Sentence Detector, you can run the detect.py script with a Japanese sentence as a command-line argument. The script will output either 1 if the sentence is derogatory or 0 if it is non-derogatory.


>Here's an example usage:

`python detect.py "今日の天気はどうですか？"`

>*This will output 0, indicating that the sentence is non-derogatory.*



## Training

If you'd like to train the model yourself using a different dataset, you can use the train.py script. The script expects a CSV file containing Japanese sentences labeled as either derogatory or non-derogatory. You can modify the script to adjust the hyperparameters and other training settings.

## Model

The model architecture used for the Japanese Derogatory Sentence Detector is the BERT (Bidirectional Encoder Representations from Transformers) model, which is a pre-trained transformer-based neural network model developed by Google. The BERT model was fine-tuned using the cl-tohoku/bert-base-japanese-whole-word-masking pre-trained model. The model was trained using PyTorch and achieved an accuracy of 99.4% on the test set. It wasn't trained on misogynistic comments, so it doesn't detect it well... It will be fixed in the V2 of this model...


## Acknowledgements

    This project is based on the Transformers library by Hugging Face: https://github.com/huggingface/transformers
    The BERT model used for this project is based on the pre-trained model by the Tohoku University NLP Lab: https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as you include the original license file.
