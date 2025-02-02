import json
import time

from enum import Enum
from pydantic import BaseModel, Field
from typing import List

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
ENTRY_LIMIT = 55000  # Limit number of hashtags to process
SOURCE_FILE = "hashtags_raw.csv"
TARGET_FILE = "hashtags_classified.csv"

PRICE_PER_1M_INPUT_TOKENS = {
    "gpt-4o-mini": 0.15,  # $0.150 per 1M input tokens
    "gpt-4o": 2.50,  # $2.50 per 1M input tokens
    "gpt-4o-2024-11-20": 2.50,  # $2.50 per 1M input tokens
    "gpt-4": 30.00,  # $30.00 per 1M input tokens
}

PRICE_PER_1M_OUTPUT_TOKENS = {
    "gpt-4o-mini": 0.60,  # $0.600 per 1M output tokens
    "gpt-4o": 10.00,  # $10.00 per 1M output tokens
    "gpt-4o-2024-11-20": 10.00,  # $10.00 per 1M output tokens
    "gpt-4": 60.00,  # $60.00 per 1M output tokens
}

MODELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-2024-11-20": "gpt-4o-2024-11-20",
    "gpt-4": "gpt-4o",
}
MODEL = MODELS["gpt-4o-mini"]

# Retry configuration
MAX_RETRIES = 5
MIN_WAIT_SECONDS = 1
MAX_WAIT_SECONDS = 60


class HashtagCategoryEnum(str, Enum):
    """Enumeration of all possible hashtag categories"""

    # Updated Classification
    ARMED_CONFLICT = "Armed Conflict/Wars/Violence"
    ANTISEMITISM = "Anti-Semitism/Anti-Zionism"
    BIODIVERSITY = "Biodiversity/Animal Conservation"
    CLIMATE = "Climate Change/Environmental Issues"
    COMEDY = "Comedy/Humor/Satire"
    CONFERENCES = "Conferences/Summits/Events"
    COVID = "COVID-19/Pandemic/Vaccines"
    CRIME = "Crime/Police/Law Enforcement/Security"
    CRYPTO = "Cryptocurrency/Blockchain/DeFi"
    CULTURAL_APPROPRIATION = "Cultural Appropriation/Ethnic Stereotypes/Minority Groups"
    CULTURE = "Culture/Arts/Traditions"
    CYBERSECURITY = "Cybersecurity and Cyber Warfare"
    DEMOCRACY = "Democracy/Rule of Law/Constitution"
    DIGITAL = "Digitalization/E-Governance/Digital Infrastructure"
    ECONOMY = "Economy/Finance/Business"
    ENERGY = "Energy/Renewables/Fossil Fuels"
    ENERGY_AND_CLIMATE = "Energy/Climate Change/Renewable Policies"
    EXTREMISM = "Extremism/Right-Wing/Left-Wing"
    FOOD = "Food/Vegan/Vegetarian/Diet"
    FUTURE_OF_WORK = "Future of Work/Digital Workspaces/Automation"
    GENDER = "Gender/Diversity/Equality"
    GEO = "Geo/Cities/Countries/Locations"
    GLOBALIZATION = "Globalization/Trade/Interconnectedness"
    HEALTH = "Health/Healthcare/Medicine"
    HOUSING = "Housing/Rent/Affordable Living"
    HUMAN_RIGHTS = "Human Rights/Abuse/Violations"
    INDUSTRY = "Industry/Manufacturing/Infrastructure"
    INNOVATION = "Innovation/Patents/Startups"
    INSTITUTIONS = "Government Institutions/Regulations/Bureaucracy"
    INTERNATIONAL = "International Relations/Geopolitics/Conflict"
    INTERNET = "Internet/Social Media/Online Communities"
    JOBS_CAREERS = "Jobs/Careers/Employment"
    LGBTQ = "LGBTQ+/LGBTQ+ Rights/Community"
    MEDIA = "TV/Media/Journalism/News Outlets"
    MENTAL_HEALTH = "Mental Health and Wellbeing"
    MILITARY = "Military/Weapons/Arms"
    MISINFORMATION = "Misinformation/Propaganda/Conspiracy Theories/Disinformation"
    MIGRATION = "Migration/Refugees/Immigration"
    MOBILITY = "Mobility/Transport/Public Infrastructure"
    MUSIC = "Music/Musicians"
    NATURE = "Nature/Wildlife/Ecology"
    OTHER = "Other/Uncategorized"
    POLITICAL = "Political Parties/Political Systems"
    POLITICAL_THEORIES = "Political Theories/Ideologies"
    POLITICIANS = "Politicians/Elected Officials"
    PUBLIC_FIGURES = "Celebrities/Artists/Influencers"
    RACISM = "Racism/Anti-Racism/Extremism"
    RELIGION = "Religion/Religious Movements/Sects"
    RENEWABLES = "Renewable Energy/Sustainability"
    SOCIAL_MOVEMENTS = "Social Movements/Protests/Activism"
    SPACE = "Space Exploration/Satellites/Space Policy"
    SPORTS = "Sports/Football"
    SURVEILLANCE = "Surveillance/Privacy/Data Protection"
    TALKSHOWS = "Talkshows/News TV/Current Affairs Programs"
    TECHNOLOGY = "Technology/AI/Robotics"
    TRANSPORT_REVOLUTION = "Transport Revolution/Public Transit/Electric Vehicles"
    UNEMPLOYMENT = "Unemployment/Economic Crisis/Recession"
    WOKENESS = "Wokeness/Identity Politics/Cancel Culture"
    WEATHER = "Weather"


class HashtagCategories(BaseModel):
    categories: List[HashtagCategoryEnum] = Field(
        description="List of hashtag categories"
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


# Retry decorator using tenacity
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=MIN_WAIT_SECONDS, max=MAX_WAIT_SECONDS),
    before_sleep=lambda retry_state: print(
        f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
    ),
)
def make_api_request(client, hashtag, model):
    """Make an API request with retry logic"""
    return client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Classify German social media hashtags into categories. Respond with JSON only.",
            },
            {
                "role": "user",
                "content": f"""Classify #{hashtag} into one or more of categories in the HashtagCategoryEnum.""",
            },
        ],
        response_format=HashtagCategories,
        temperature=0.0,
    )


def classify_hashtags(input_file: str, output_file: str):
    """Main function to classify hashtags and track costs."""

    # Initialize OpenAI client and load data
    client = OpenAI()
    data = pd.read_csv(input_file)
    if ENTRY_LIMIT:
        data = data.head(ENTRY_LIMIT)

    # Initialize tracking variables
    classifications = []
    total_input_tokens = 0
    total_output_tokens = 0

    # Process each hashtag
    print(f"Processing {len(data)} hashtags...")
    for index, row in data.iterrows():
        hashtag = row["hashtag"]
        print(f"\nProcessing {index + 1}/{len(data)}: #{hashtag}")

        try:
            # Make API request with retry logic
            response = make_api_request(client, hashtag, MODEL)

            # Parse response
            try:
                result = json.loads(response.choices[0].message.content)
                categories = result.get("categories", ["Other"])
                confidence = result.get("confidence", 0.0)
            except:
                categories = ["Other"]
                confidence = 0.0

            # Track tokens
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

            # Store results
            classifications.append(
                {
                    "hashtag": hashtag,
                    "categories": ";".join(categories),
                    "confidence": confidence,
                }
            )

            # Print progress
            print(f"  Categories: {', '.join(categories)}")
            print(f"  Confidence: {confidence:.2f}")

        except Exception as e:
            print(f"Failed to process hashtag '{hashtag}' after all retries: {str(e)}")
            # Store failure result
            classifications.append(
                {"hashtag": hashtag, "categories": "Other", "confidence": 0.0}
            )

        time.sleep(0.1)  # Small delay to avoid rate limits

    # Create DataFrame with classifications and merge with original data
    classifications_df = pd.DataFrame(classifications)
    result = pd.merge(data, classifications_df, on="hashtag", how="left")

    # Save results
    result.to_csv(output_file, index=False)

    # Print summary
    print("\nClassification Complete!")
    print(f"Total Input Tokens: {total_input_tokens:,}")
    print(f"Total Output Tokens: {total_output_tokens:,}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    classify_hashtags(SOURCE_FILE, TARGET_FILE)
