from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Fake News Detection Model (Hugging Face)
classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

# LLaMA / text generation model (lighter alternative if needed)
generator = pipeline(
    "text-generation",
    model="distilgpt2"  # replace with LLaMA if GPU available
)

# Detect fake news
def detect_news(text):
    result = classifier(text)[0]
    label = result['label']
    score = round(result['score'] * 100, 2)
    return label, score

# Generate explanation
def explain_news(text):
    prompt = f"Explain in simple points why this news may be fake or real:\n{text}\nAnswer:"
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]

    label, score = detect_news(text)
    explanation = explain_news(text)

    return jsonify({
        "label": label,
        "confidence": score,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)