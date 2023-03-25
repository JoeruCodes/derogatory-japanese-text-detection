# Japanese Derogatory Sentence Detector

This project contains a neural network model trained to detect if a given Japanese sentence is derogatory or non-derogatory. The model was trained using a dataset of Japanese sentences labeled as either derogatory or non-derogatory.



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



## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as you include the original license file.
