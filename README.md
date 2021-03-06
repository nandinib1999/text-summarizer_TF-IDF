# Extractive Summarization of text using TF-IDF

![altext](https://miro.medium.com/max/1064/1*GIVviyN9Q0cqObcy-q-juQ.png)

This program accepts a text file using cmd argument --filepath and it returns an extractive summary of the text. The summary may not be completely accurate but it manages to give a fair idea to the user about the text.

## Usage

There's one commandline argument that needs to passed while running the script i.e. **--filepath**. It accepts the path to the txt file which needs to be summarized.

In Anaconda Prompt or CMD, run
```
python text_summarizer.py --filepath="articles/article.txt"
```
I have deployed this project as an API on Heroku. 
```
https://summarizertxt.herokuapp.com/summarize
```
The text to be summarized can be sent as a POST request. An example using Python can be found below:
```
import requests
url = "https://summarizertxt.herokuapp.com/summarize"
data = {"text":text}
x = requests.post(url, data).json()
```
