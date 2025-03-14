import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, errors

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def load_csv_to_mongo(csv_file_path: str):
    """
    Loads CSV data into MongoDB.
    Each row in the CSV becomes a separate document in the specified collection.
    """
    try:
        # Read CSV using pandas
        df = pd.read_csv(csv_file_path)
        data = df.to_dict("records")
        if not data:
            print("CSV file is empty!")
            return

        # Connect to MongoDB and insert data
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        result = db[COLLECTION_NAME].insert_many(data)
        print(f"Inserted {len(result.inserted_ids)} documents into MongoDB collection '{COLLECTION_NAME}'.")
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} does not exist.")
    except errors.ConnectionFailure as e:
        print(f"MongoDB connection error: {e}")
    except Exception as e:
        print(f"An error occurred while loading CSV data: {e}")

if __name__ == "__main__":
    # For direct execution, you can specify the CSV file path here.
    csv_path = "sample_data.csv"
    load_csv_to_mongo(csv_path)



