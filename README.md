# SentimentMaster

SentimentMaster is a Python application designed for sentiment analysis. It utilizes machine learning techniques to analyze the sentiment of text data. The sentiment analysis model is trained using the IMDB reviews dataset obtained from Kaggle. The application preprocesses the data and deploys the trained model for sentiment analysis.

## Installation

To install SentimentMaster, follow these steps:

1. Clone the repository:

```
git clone https://github.com/your_username/SentimentMaster.git
```

2. Navigate to the project directory:

```
cd SentimentMaster
```

3. Download the IMDB reviews dataset from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

4. Extract the downloaded dataset and move it to the `data` folder within the SentimentMaster directory.

5. Install the required Python dependencies:

```
pip install -r requirements.txt
```

## Usage

To use SentimentMaster, execute the following command:

```
python app.py
```

This will start the application and perform sentiment analysis on the provided text data.

## Preprocessing

Before training the sentiment analysis model, the data undergoes preprocessing. This includes steps such as tokenization, removing stop words, and converting text to numerical representation suitable for machine learning algorithms.

## Deployment

Once the model is trained, it is deployed within the SentimentMaster application. Users can input text data, and the deployed model will predict the sentiment of the input text.

## Contributing

Contributions to SentimentMaster are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on GitHub.

## Acknowledgments

- The SentimentMaster application utilizes the IMDB reviews dataset from Kaggle.
- Special thanks to the contributors of the Python libraries used in this project.