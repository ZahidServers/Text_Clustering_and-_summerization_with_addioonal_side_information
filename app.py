from flask import Flask, request, render_template, redirect, url_for
import fitz  # PyMuPDF
from transformers import pipeline
from datetime import datetime
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, silhouette_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from werkzeug.utils import secure_filename
from sklearn.cluster import BisectingKMeans
import numpy as np
import sqlite3
import logging
import time
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Load the summarization model
summarizer = pipeline('summarization')
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',filename='app.log',  filemode='w')
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, filename TEXT, size INTEGER, upload_date TEXT, text TEXT, summary TEXT, cluster INTEGER)''')
    conn.commit()
    conn.close()
def insert_document(filename, size, upload_date, text, summary, cluster):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO documents (filename, size, upload_date, text, summary, cluster) VALUES (?, ?, ?, ?, ?, ?)",
              (filename, size, upload_date, text, summary, cluster))
    conn.commit()
    conn.close()
def get_all_documents():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM documents")
    rows = c.fetchall()
    conn.close()
    return rows
init_db()

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    try:
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
    return text

def determine_optimal_clusters(X):
    silhouette_scores = []
    for k in range(2, min(11, X.shape[0])):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
    return optimal_k

"""def bisecting_kmeans(data, k):
    # Initialize the clusters with the entire dataset as one cluster
    clusters = [data]
    while len(clusters) < k:
        try:
            # Find the largest cluster to bisect
            largest_cluster_idx = max(range(len(clusters)), key=lambda i: clusters[i].shape[0])
            largest_cluster = clusters.pop(largest_cluster_idx)
            # If the largest cluster has fewer than 2 samples, add it back and break the loop
            if largest_cluster.shape[0] < 2:
                clusters.append(largest_cluster)
                break
            # Bisect the largest cluster using KMeans with 2 clusters
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            new_labels = km.fit_predict(largest_cluster)
            # Create new clusters from the bisected largest cluster
            cluster_1 = largest_cluster[new_labels == 0, :]  # Ensure correct indexing for sparse matrices
            cluster_2 = largest_cluster[new_labels == 1, :]  # Ensure correct indexing for sparse matrices
            # Add the new clusters to the list of clusters
            clusters.extend([cluster_1, cluster_2])
        except Exception as e:
            logging.error(f"Error in bisecting K-means: {e}")
    return clusters"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logging.debug("File saved: %s", file_path)
        # Extract text from the PDF
        text = extract_text_from_pdf(file_path)
        if text is None:
            return render_template('error.html', message="Cant extract No Text to extract. Check PDF Format properly")
        logging.debug("Text extracted: %s", text[:100])
        # Summarize the text     
        try:
            # Summarize the text
            summary = summarizer(text, max_length=500, min_length=50, do_sample=False)[0]['summary_text']
            logging.debug("Summary created: %s", summary[:100])
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return render_template('error.html', message="An error occurred during text summarization: " + str(e))
        # Collect additional information
        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_size = os.path.getsize(file_path)
        file_meta_data = {
            "filename": file.filename,
            "size": os.path.getsize(file_path),
            "upload_date": upload_date,
            "links": [url_for('static', filename=file.filename)],
        }
        # Data collection and preparation
        # Insert document data into the database
        insert_document(filename, file_size, upload_date, text, summary, cluster=None)
        # Retrieve all documents from the database
        documents = get_all_documents()
        data = {
            "text": text,
            "summary": summary,
            "metadata": file_meta_data,
        }
        # Convert to DataFrame for clustering and classification
        df = pd.DataFrame(documents, columns=['id', 'filename', 'size', 'upload_date', 'text', 'summary', 'cluster'])
        # Vectorize text data
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['text'])
        logging.debug("Text vectorized")
        # Bisecting K-means clustering
        try:
            k=determine_optimal_clusters(X)
            logging.debug(f'Optimal Clusters: {k}')
        except:
            k=2
            logging.debug(f'Optimal Clusters: {k}')
        try:
            bkm = BisectingKMeans(n_clusters=k, random_state=42)
            labels = bkm.fit_predict(X)
            clusters = bkm.cluster_centers_  # Example with k=2 clusters
        except Exception as e:
            logging.error(f"Error in bisecting K-means: {e}")
            clusters = []
        if len(clusters) > 1:
            # Assign labels based on clusters
            for i in range(len(clusters)):
                cluster_indices = np.where(labels == i)[0]
                df.loc[cluster_indices, 'cluster'] = i
            # Update cluster information in the database
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            for index, row in df.iterrows():
                c.execute("UPDATE documents SET cluster=? WHERE id=?", (row['cluster'], row['id']))
                logging.info(f"ID: {row['id']} , Cluster: {row['cluster']}")
            conn.commit()
            conn.close()
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)
            logging.info(f"X_train is {X_train}")
            logging.info(f"y_train is {y_train}")
            logging.info(f"X_test is {X_test}")
            logging.info(f"y_test is {y_test}")
            # Train a classifier
            classifier = MultinomialNB()
            classifier.fit(X_train, y_train)
            # Predictions
            y_pred = classifier.predict(X_test)
            logging.info(f"Y pred is {y_pred}")
            # Performance metrics
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            results = {
                "precision": precision,
                "recall": recall,
            }
        else:
            logging.info("less erorr less than 1 cluster")
            results = {
                "precision": None,
                "recall": None,
            }
        end_time = time.time()
        logging.debug(f"Total execution time: {end_time - start_time} seconds")
        return render_template('summary.html', summary=summary, metadata=file_meta_data, results=results)
# Route for displaying clusters and associated documents
@app.route('/clusters')
def clusters():
    # Retrieve documents from the database
    documents = get_all_documents()
    # Convert to DataFrame
    df = pd.DataFrame(documents, columns=['id', 'filename', 'size', 'upload_date', 'text', 'summary', 'cluster'])
    # Group documents by cluster
    grouped_clusters = df.groupby('cluster')
    # Prepare data to pass to template
    cluster_data = []
    for cluster, group in grouped_clusters:
        cluster_data.append({
            'cluster': cluster,
            'documents': group[['filename', 'summary']].to_dict(orient='records')
        })
    return render_template('clusters.html', clusters=cluster_data)

if __name__ == '__main__':
    app.run(debug=True, port=8888)
