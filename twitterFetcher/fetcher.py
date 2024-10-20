import tweepy
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Twitter API credentials using environment variables
consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# Check if any credentials are missing
if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
    raise ValueError("Missing one or more Twitter API credentials in the .env file.")

# Authenticate with the API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Function to fetch tweets from a user
def fetch_tweets(username, max_tweets=1000):
    tweets = []
    try:
        for tweet in tweepy.Cursor(api.user_timeline, screen_name=username, tweet_mode='extended').items(max_tweets):
            tweets.append({
                'user': username,
                'content': tweet.full_text,
                'date_time': tweet.created_at,
                'id': tweet.id
            })
    except Exception as e:
        print(f"Error fetching tweets for {username}: {e}")
    return tweets

# List of usernames to fetch tweets from
usernames = ['kmitsotakis']  # Add up to 20-50 users here

# Collect data for each user
all_tweets = []
for username in usernames:
    user_tweets = fetch_tweets(username, max_tweets=1000)
    all_tweets.extend(user_tweets)
    print(f"Collected {len(user_tweets)} tweets for {username}")

# Convert to DataFrame and save as CSV
df = pd.DataFrame(all_tweets)
df.to_csv('twitter_dataset.csv', index=False)
print("Saved dataset to twitter_dataset.csv")
