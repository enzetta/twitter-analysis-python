from transformers import pipeline
from colorama import Fore, Style, init

# Initialize colorama for color support
init(autoreset=True)

# Define models for sentiment analysis and toxicity detection
TOXICITY_MODEL = "textdetox/xlmr-large-toxicity-classifier"
SENTIMENT_MODEL = "oliverguhr/german-sentiment-bert"

# Load the sentiment and toxicity classification pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
toxicity_pipeline = pipeline("text-classification", model=TOXICITY_MODEL)

# Define color mapping for sentiment results
SENTIMENT_COLORS = {
    "positive": Fore.GREEN,
    "neutral": Fore.YELLOW,
    "negative": Fore.RED
}

# Define color mapping for toxicity results
TOXICITY_COLORS = {
    "non-toxic": Fore.GREEN,
    "neutral": Fore.YELLOW,
    "toxic": Fore.RED
}


def analyze_tweet(tweet_text):
    """Analyze sentiment and toxicity of a tweet and print results with color coding."""

    sentiment_result = sentiment_pipeline(tweet_text)[0]
    toxicity_result = toxicity_pipeline(tweet_text)[0]

    sentiment_label = sentiment_result['label']
    toxicity_label = toxicity_result['label']

    sentiment_score = sentiment_result['score']
    toxicity_score = toxicity_result['score']

    sentiment_color = SENTIMENT_COLORS.get(sentiment_label, Fore.WHITE)
    toxicity_color = TOXICITY_COLORS.get(toxicity_label, Fore.WHITE)

    print(
        f"{Style.BRIGHT}{Fore.CYAN}──────────────────────────────────────────")
    print(f"📝 {Style.BRIGHT}Tweet: {Fore.WHITE}{tweet_text}\n")

    print(
        f"{Style.BRIGHT}📊 Sentiment Analysis: {sentiment_color}{sentiment_label.upper()} ({sentiment_score:.2%})"
    )
    print(
        f"{Style.BRIGHT}☣ Toxicity Classification: {toxicity_color}{toxicity_label.upper()} ({toxicity_score:.2%})"
    )

    print(
        f"{Style.BRIGHT}{Fore.CYAN}──────────────────────────────────────────\n"
    )


# Negative Example (Hate Speech / Offensive)
analyze_tweet(
    "Die AfD ist voll mit Nazis und ich würde sie am liebsten alle abschieben."
)

# Positive Example (Polite and Friendly)
analyze_tweet("Das ist aber ein toller Ort. Gerne komme ich wieder.")

# Neutral Example (Informative but Unemotional)
analyze_tweet("Die Bundestagswahl findet am 26. September statt.")

# Sarcasm/Irony Example (Tricky for Sentiment Analysis)
analyze_tweet(
    "Oh toll, noch eine neue Steuererhöhung... genau das hat uns gefehlt.")

# Controversial Example (Mixed Sentiment and Possible Toxicity)
analyze_tweet("Politiker lügen sowieso alle! Man kann keinem vertrauen.")

# Enthusiastic Example (Overtly Positive)
analyze_tweet("Das Konzert war einfach unglaublich! Beste Stimmung ever! 🎶✨")

# Ambiguous Example (Could be Positive or Negative Depending on Context)
analyze_tweet("Naja, das Essen hier ist schon... besonders.")

# tweet examples
analyze_tweet(
    "@RND_de @spdde Ihr habt sie echt nicht mehr alle. Menschen in diesem Land interessieren Euch einen dampfenden Haufen Scheiße"
)

analyze_tweet(
    "Kann man nur noch sagen #HaltDieFresseBild #HaltDieFresseSpringerPresse #HaltDieFresseAfD"
)

analyze_tweet(
    "@Sabine60451919 @ABaerbock Glaubst du wir lassen uns erpressen? Ich bin jetzt für 2G bundesweit, dann können Querdenker schön zuhause querdenken und wenn sie in der Öffentlichkeit nerven gibt es aufs Maul #QuerdenkerSindTerroristen"
)

# for precision, recall, and accuracy 
# https://huggingface.co/textdetox/xlmr-large-toxicity-classifier
