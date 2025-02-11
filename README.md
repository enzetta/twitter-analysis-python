
# **Twitter Sentiment and Toxicity Analysis - README**

## **Project Overview**
This project analyzes Twitter data for **sentiment and toxicity classification** using **Machine Learning (ML) models**. The goal is to detect **emotions, toxicity levels, and sentiment polarity** in tweets. The results are stored in **Google BigQuery** for further analysis.

The project includes:
- **Toxicity & Sentiment Classification** using **transformers**.
- **BigQuery Integration** for **storing and retrieving data**.
- **Hashtag Categorization** using **GPT models**.
- **Hyperparameter Testing** for optimizing **ML inference speed**.
- **Parallel Processing Pipelines** for handling large datasets.

---

## **⚙️ Installation & Setup**
### **1️. Setup Virtual Environment**
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### **2️. Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️. Authenticate with Google Cloud**
Ensure you have a **Google Cloud service account** and set up authentication:
```sh
export GOOGLE_APPLICATION_CREDENTIALS=.secrets/service-account.json
```

---

## **Running the Programs**
### **1️. Sentiment & Toxicity Classification**
The script **\`toxicity.py\`** runs sentiment and toxicity classification on predefined tweets.

**Run it with:**
```sh
python toxicity/toxicity.py
```
This will output results with **color-coded sentiment and toxicity analysis**.

---

### **2️. Batch Processing with Google BigQuery**
To process **thousands of tweets in parallel**, use:
```sh
python toxicity/predict_v3.py
```
🔹 **What it does:**
- Fetches **tweets from BigQuery**.
- Analyzes **sentiment & toxicity** using **deep learning models**.
- Stores the results **back in BigQuery** for analysis.

---

### **3. Hashtag Categorization**
To **classify hashtags** into relevant topics, run:
```sh
python hashtags/classify_hashtags.py
```
🔹 **What it does:**
- Uses this query as input dataset `get_hashtag_counts`
- Reads **hashtags from \`hashtags_raw.csv\`**.
- Uses **GPT-based classification** for **semantic analysis**.
- Outputs results to **\`hashtags_classified.csv\`**.

#### **Format for BigQuery**
To prepare the classified hashtags for BigQuery:
```sh
python hashtags/format_for_bigquery.py
```
This reformats the CSV to ensure **correct column types**.

---

### **4. Hyperparameter Optimization**
To test **different batch processing configurations**, run:
```sh
python toxicity/hyperparameter_tester.py
```
🔹 **What it does:**
- Tests **multiple configurations** to find the **fastest ML inference setup**.
- Saves **performance results** to a JSON file.

---

## **Querying Results in BigQuery**
To analyze sentiment and toxicity **grouped by week**, use:
```sql
SELECT 
  DATE_TRUNC(recorded_at, WEEK(MONDAY)) AS week_start,
  COUNT(tweet_id) AS total_tweets,
  AVG(toxicity_score) AS avg_toxicity_score
FROM `grounded-nebula-408412.twitter.backup_tweet_sentiment_analysis`
GROUP BY week_start
ORDER BY week_start DESC;
```

---

## **Folder Structure**
```
twitter-analysis-python/
├── predictions/                      # Contains tweet predictions
│   ├── predictions.json              # Example JSON output
├── hashtags/                         # Hashtag classification scripts
│   ├── classify_hashtags.py          # Categorizes hashtags
│   ├── format_for_bigquery.py        # Formats CSV for BigQuery import
│   ├── get_hashtag_counts.sql        # Query to get hashtag statistics
├── toxicity/                         # Sentiment & toxicity processing
│   ├── toxicity.py                   # Runs single tweet analysis 
│   ├── predict_v3.py                 # Batch prediction processing
│   ├── bigquery_client.py            # Handles BigQuery operations
│   ├── hyperparameter_tester.py      # ML performance testing
├── requirements.txt                  # Python dependencies
├── README.md                         # This documentation
```

---

## ** Additional Notes**
- **BigQuery is used** for scalable data handling.
- **Color-coded console output** improves interpretability.
- **Transformers used:** \`xlmr-large-toxicity-classifier\`, \`german-sentiment-bert\`.
- **Error handling** is implemented with retry mechanisms.

---

## ** Final Notes for Professors**
This project demonstrates:
✔ **Big Data Processing** with **Google BigQuery**  
✔ **Machine Learning for Sentiment & Toxicity Analysis**  
✔ **Efficient Parallel Processing** with **batch inference**  
✔ **Semantic NLP Classification** with **GPT-based models**  
✔ **Optimization Techniques** using **hyperparameter tuning**  

**For any questions, please refer to the comments inside the scripts!** 🚀🎓
