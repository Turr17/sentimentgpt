import pandas as pd
from pymongo import MongoClient

# Baca CSV
df = pd.read_csv('D:/Kampus/PA/PA FATUR/Sistem/sentiment_balanced_text.csv')

# Koneksi ke MongoDB
client = MongoClient(
    'mongodb://fatur21si:fatur21si@cluster0-shard-00-00.wiqdc.mongodb.net:27017,cluster0-shard-00-01.wiqdc.mongodb.net:27017,cluster0-shard-00-02.wiqdc.mongodb.net:27017/?replicaSet=atlas-11dt48-shard-0&ssl=true&authSource=admin'
)

db = client['db_pa']
inputan_collection = db['data_inputan']

# Tambahkan field source
df['source'] = 'initial'

if inputan_collection.count_documents({'source': 'initial'}) == 0:
    inputan_collection.insert_many(df[['text', 'sentiment', 'source']].to_dict(orient='records'))
    print("Data awal berhasil disimpan ke MongoDB.")
else:
    print("Data awal sudah ada.")