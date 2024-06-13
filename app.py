from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load models using relative paths
classification_model = joblib.load(os.path.join(base_dir, 'text_classification_model.pkl'))
lda_model = joblib.load(os.path.join(base_dir, 'lda_model.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(base_dir, 'tfidf_vectorizer.pkl'))

# Define topic names
topic_names = {
    0: "Debt and Credit Reporting",
    1: "Debt Collection and Loans",
    2: "Mortgage and Loan Payments",
    3: "Credit Reporting Issues",
    4: "Credit Card and Payment Issues",
    5: "Debt Collection Practices",
    6: "Identity Theft and Fraud",
    7: "Fraudulent Charges and Disputes",
    8: "Bank Account and Fraud Issues",
    9: "PNC Consumer Issues"
}

@app.route('/')
def home():
    return "Consumer Inquiry and Complaint Analysis API"

@app.route('/classify', methods=['POST'])
def classify_complaint():
    complaint = request.json.get('complaint')
    if not complaint:
        return jsonify({'error': 'No complaint provided'}), 400
    prediction = classification_model.predict([complaint])
    return jsonify({'complaint_type': prediction[0]})

@app.route('/topics', methods=['POST'])
def analyze_complaints():
    complaints = request.json.get('complaints')
    if not complaints:
        return jsonify({'error': 'No complaints provided'}), 400
    transformed_data = tfidf_vectorizer.transform(complaints)
    topics = lda_model.transform(transformed_data)
    topic_results = []
    for topic_distribution in topics:
        topic_result = {topic_names[i]: prob for i, prob in enumerate(topic_distribution)}
        topic_results.append(topic_result)
    return jsonify(topic_results)

if __name__ == '__main__':
    app.run(debug=True)
