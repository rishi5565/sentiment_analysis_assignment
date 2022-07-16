
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import os
from src.functions import get_sentiment

webapp_root = "webapp"
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, template_folder=template_dir)
CORS(app)



@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    data = request.json['data']
    result = get_sentiment(data)
    return jsonify({ "text" : result})



if __name__ == "__main__":
    app.run()
    

    ####### POSTMAN API TESTING #######
    # if request.method == "POST":
    #     req = request.json
    #     sentence = req["sentence"]
    #     sentiment = get_sentiment(sentence)
    #     return jsonify(sentiment)



