import logging
import pickle
import pandas as pd
import flask
import json
from flask_restful import reqparse
from flask import Flask, request, jsonify, Response, redirect, url_for, render_template
from preprocessing.tokenize_helper import tokenizer

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Initialize the Flask application
application = Flask(__name__)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument("comment_path")


def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp


@application.route("/")
def index():
    return render_template("predict.html")


@application.route("/predict", methods=["POST", "GET"])
def predict():
    # Load in best model
    model = pickle.load(open("model_artifacts/logistic_m1.pickle", "rb"))
    cl = pd.read_csv("intermediate_data/code_lookup.csv")
    t50 = pd.read_csv("intermediate_data/diagnoses_by_count_50.csv")
    code_lookup = t50.merge(cl, how="left", on="diagnosis")

    if request.method == "POST":
        # Load in JSON
        json_request = request.form["drug_text"]
        if not json_request:
            return Response("No json provided.", status=400)
        X_test = tokenizer(json_request)
        # X_test = json_request['text']
        if X_test is None:
            return Response("No text provided.", status=400)
        else:
            # return flask.jsonify(X_test)
            predictions = model.predict(X_test).astype(str).tolist()
            predicted = str(
                code_lookup[code_lookup["diagnosis"] == predictions[0]][
                    "long_title"
                ].values[0]
                + " - "
                + predictions[0]
            )
            predicted_prob = pd.DataFrame(model.predict_proba(X_test))
            predicted_prob["pred_value"] = predicted_prob.max(axis=1)
            output_dict = {
                "input_text": " ".join(X_test),
                "pred_prob": predicted_prob["pred_value"].values.round(3).tolist(),
                "pred_value": predicted,
            }

            results = pd.DataFrame(output_dict)
            results2 = results[:1]
            output = results2.to_json(orient="records")
            parsed = json.loads(output)

            return redirect(url_for("results", response_text=parsed))

    else:
        return render_template("predict.html")


@application.route("/results/<response_text>")
def results(response_text):
    json_response = json.loads(str(json.dumps(eval(response_text))))
    words_used = json_response["input_text"]
    prob_pred = json_response["pred_prob"]
    pred_value = json_response["pred_value"]
    return render_template(
        "response.html",
        words_used=words_used,
        prob_pred=prob_pred,
        pred_value=pred_value,
    )


if __name__ == "__main__":
    application.run(debug=True, port="5000", use_reloader=True)
