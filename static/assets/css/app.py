from flask import Flask, request, render_template, redirect, url_for, send_file, flash, session, jsonify
from pymongo import MongoClient
import pandas as pd
import os, re, string, pickle
import matplotlib.pyplot as plt
import nltk
from werkzeug.utils import secure_filename
from datetime import datetime
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from googletrans import Translator
import emoji

app = Flask(__name__)
app.secret_key = 'fatur21sipcr' 

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Koneksi MongoDB
client = MongoClient(
    'mongodb://fatur21si:fatur21si@cluster0-shard-00-00.wiqdc.mongodb.net:27017,cluster0-shard-00-01.wiqdc.mongodb.net:27017,cluster0-shard-00-02.wiqdc.mongodb.net:27017/?replicaSet=atlas-11dt48-shard-0&ssl=true&authSource=admin'
)
db = client['db_pa']
inputan_collection = db['data_inputan']

# Load model & vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Inisialisasi translator Google
translator = Translator()

# Inisialisasi stopwords dan stemmer Bahasa Indonesia
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())
stemmer = StemmerFactory().create_stemmer()

# Fungsi untuk membersihkan teks dan melakukan preprocessing lengkap
def preprocess_text(text):
    if pd.isna(text):
        return ""

    # Ubah ke lowercase
    text = text.lower()

    # Hilangkan mention, hashtag, dan URL
    text = re.sub(r'@[\w_]+', '', text)         # hapus mention
    text = re.sub(r'#\w+', '', text)            # hapus hashtag
    text = re.sub(r'http\S+|www\S+', '', text)  # hapus URL

    # Hapus angka dan tanda baca
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Hapus emoji
    text = emoji.replace_emoji(text, replace='')

    # Hilangkan spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenisasi
    tokens = word_tokenize(text)

    # Hapus stopwords Bahasa Indonesia
    filtered_tokens = [word for word in tokens if word not in stopwords]

    # Stemming Bahasa Indonesia
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Gabungkan kembali menjadi kalimat
    cleaned_text = ' '.join(stemmed_tokens)

    return cleaned_text

@app.route('/')
def index():
    # Ambil data sentimen dari MongoDB
    all_data = inputan_collection.find({}, {'_id': 0, 'sentiment': 1})
    sentiment_count = {'positif': 0, 'negatif': 0, 'netral': 0}

    for row in all_data:
        label = row.get('sentiment', '').lower()
        if label in sentiment_count:
            sentiment_count[label] += 1

    # Tampilkan WordCloud
    all_data = list(inputan_collection.find({}, {'_id': 0, 'text': 1, 'sentiment': 1}))
    words_per_sentiment = {'positif': [], 'negatif': [], 'netral': []}

    for item in all_data:
        sentimen = item.get('sentiment', '').lower()
        text = item.get('text', '')  # sudah teks bersih
        if sentimen in words_per_sentiment and text:
            tokens = text.split()
            words_per_sentiment[sentimen].extend(tokens)

    # Buat folder static jika belum ada
    output_folder = os.path.join('static', 'wordclouds')
    os.makedirs(output_folder, exist_ok=True)

    wordcloud_images = {
        'positif': 'assets/wordcloud/positif.png',
        'negatif': 'assets/wordcloud/negatif.png',
        'netral': 'assets/wordcloud/netral.png'
    }

    sentiment_count = {
        'positif': inputan_collection.count_documents({'sentiment': 'positif'}),
        'negatif': inputan_collection.count_documents({'sentiment': 'negatif'}),
        'netral': inputan_collection.count_documents({'sentiment': 'netral'}),
    }

    for sentimen, words in words_per_sentiment.items():
        if not words:
            continue
        text = ' '.join(words)
        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
        filepath = os.path.join(output_folder, f"{sentimen}.png")
        wc.to_file(filepath)
        wordcloud_images[sentimen] = f"wordclouds/{sentimen}.png"
        
    total_all = sum(sentiment_count.values())
    if total_all > 0:
        sentiment_percentage = {
            k: round((v / total_all) * 100, 2) for k, v in sentiment_count.items()
        }
    else:
        sentiment_percentage = {'positif': 0, 'negatif': 0, 'netral': 0}

    top_words = {}
    for sentimen, words in words_per_sentiment.items():
        if words:
            counter = Counter(words)
            top_words[sentimen] = counter.most_common()
        else:
            top_words[sentimen] = []

    return render_template('index.html', prediction=None, sentiment_count=sentiment_count, sentiment_percentage=sentiment_percentage, wordcloud_images=wordcloud_images, top_words=top_words)

@app.route('/sentiment_chart_data', methods=['GET'])
def sentiment_chart_data():
        if 'last_predicted_sentiments' in session:
            sentiment_counts = session['last_predicted_sentiments']
            return jsonify(sentiment_counts)
        else:
            return jsonify({})

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                flash(f"Error membaca file CSV: {e}", 'danger')
                return redirect(request.url)

            if 'text' not in df.columns:
                flash('CSV harus memiliki kolom bernama "text"', 'danger')
                return redirect(request.url)

            df.dropna(subset=['text'], inplace=True)
            df = df[df['text'].apply(lambda x: isinstance(x, str))]

            # Preprocessing
            df['text'] = df['text'].astype(str).apply(preprocess_text)  # overwrite text jadi cleaned text

            # Ambil semua teks yang sudah ada di database
            existing_texts = set(doc['text'] for doc in inputan_collection.find({}, {'text': 1}))

            # Ambil data baru yang belum ada
            new_data = df[~df['text'].isin(existing_texts)].copy()
            dupe_count = len(df) - len(new_data)

            if new_data.empty:
                flash('Semua data yang Anda unggah sudah ada di database dan tidak disimpan ulang.', 'info')
                return render_template('upload.html', table_html=None)

            # Prediksi
            X_new = vectorizer.transform(new_data['text'])
            new_data['sentiment'] = model.predict(X_new)

            sentiment_counts = Counter(new_data['sentiment'])  # dari hasil prediksi
            session['last_predicted_sentiments'] = dict(sentiment_counts)

            # Simpan ke MongoDB dengan teks yang sudah clean
            inputan_collection.insert_many(new_data[['text', 'sentiment']].to_dict(orient='records'))

            # Simpan ke file hasil
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"hasil_prediksi_{timestamp}.csv"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            new_data[['text', 'sentiment']].to_csv(output_path, index=False)

            if dupe_count > 0:
                flash(f"{len(new_data)} data berhasil ditambahkan. {dupe_count} data duplikat diabaikan.", 'warning')
            else:
                flash(f"Semua {len(new_data)} data berhasil ditambahkan!", 'success')

            table_html = new_data[['text', 'sentiment']].to_html(classes='table table-bordered table-striped', index=False, border=0)
            return render_template('upload.html', table_html=table_html, output_filename=output_filename)

        else:
            flash('Format file tidak didukung. Harap unggah file CSV.', 'danger')
            return redirect(request.url)

    return render_template('upload.html', table_html=None)

@app.route('/download/<filename>')
def download(filename):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(output_path, as_attachment=True)

@app.route('/hasil')
def hasil():
    filter_sentiment = request.args.get('sentiment')
    query = {}
    if filter_sentiment:
        query = {'sentiment': filter_sentiment}

    data = list(inputan_collection.find(query, {'_id': 0}))

    # Ambil data sentimen dari MongoDB
    all_data = inputan_collection.find({}, {'_id': 0, 'sentiment': 1})
    sentiment_count = {'positif': 0, 'negatif': 0, 'netral': 0}

    for row in all_data:
        label = row.get('sentiment', '').lower()
        if label in sentiment_count:
            sentiment_count[label] += 1

    sentiment_count = {
        'positif': inputan_collection.count_documents({'sentiment': 'positif'}),
        'negatif': inputan_collection.count_documents({'sentiment': 'negatif'}),
        'netral': inputan_collection.count_documents({'sentiment': 'netral'}),
    }

    total_all = sum(sentiment_count.values())
    if total_all > 0:
        sentiment_percentage = {
            k: round((v / total_all) * 100, 2) for k, v in sentiment_count.items()
        }
    else:
        sentiment_percentage = {'positif': 0, 'negatif': 0, 'netral': 0}

    return render_template('hasil-klasifikasi.html', data=data, selected_filter=filter_sentiment, sentiment_count=sentiment_count, sentiment_percentage=sentiment_percentage)

if __name__ == '__main__':
    app.run(debug=True)