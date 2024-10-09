import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load datasets
calls_df = pd.read_csv('calls.csv')
customers_df = pd.read_csv('customers.csv')
reasons_df = pd.read_csv('reasons.csv')
sentiment_df = pd.read_csv('sentiment_statistics.csv')
test_df = pd.read_csv('test.csv')

# Check for missing columns
print("Columns in calls_df:", calls_df.columns.tolist())
print("Columns in sentiment_df:", sentiment_df.columns.tolist())

# If average_sentiment and silence_percent_average are not in calls_df, merge sentiment statistics
if 'average_sentiment' not in calls_df.columns or 'silence_percent_average' not in calls_df.columns:
    calls_df = calls_df.merge(sentiment_df[['call_id', 'average_sentiment', 'silence_percent_average']], on='call_id', how='left')

# Calculate AHT
calls_df['call_duration'] = pd.to_datetime(calls_df['call_end_datetime']) - pd.to_datetime(calls_df['agent_assigned_datetime'])
AHT = calls_df['call_duration'].dt.total_seconds().mean()
print("Average Handle Time (AHT):", AHT, "seconds")

# Calculate AST
calls_df['wait_time'] = pd.to_datetime(calls_df['agent_assigned_datetime']) - pd.to_datetime(calls_df['call_start_datetime'])
AST = calls_df['wait_time'].dt.total_seconds().mean()
print("Average Speed to Answer (AST):", AST, "seconds")

# AHT by Agent
AHT_by_agent = calls_df.groupby('agent_id')['call_duration'].mean()
print("AHT by Agent:\n", AHT_by_agent)

# Sentiment Analysis for AHT
if 'average_sentiment' in calls_df.columns and 'silence_percent_average' in calls_df.columns:
    X = calls_df[['average_sentiment', 'silence_percent_average']]
    y = calls_df['call_duration'].dt.total_seconds()

    # Drop rows with NaN values in X or y
    valid_indices = X.notna().all(axis=1) & y.notna()
    X_valid = X[valid_indices]
    y_valid = y[valid_indices]

    model = LinearRegression().fit(X_valid, y_valid)
    print("AHT Coefficients:", model.coef_)

# Frequent Call Reasons
frequent_reasons = calls_df['call_transcript'].str.extractall(r'(\w+)')  # Example to extract words
reason_counts = frequent_reasons[0].value_counts()
print("Frequent Call Reasons:\n", reason_counts)

# Categorizing Call Reasons
reason_df = reasons_df.groupby('primary_call_reason').agg({'call_id': 'count'}).reset_index()
print("Call Reasons Count:\n", reason_df)

# Self-Service and IVR Optimization
vectorizer = CountVectorizer(stop_words='english', max_features=50)
X_transcript = vectorizer.fit_transform(calls_df['call_transcript'].fillna(''))  # Fill NaN with empty string
frequent_words = vectorizer.get_feature_names_out()
print("Frequent words in transcripts:\n", frequent_words)

# Call Reason Prediction
if 'call_transcript' in calls_df.columns:
    X_train, X_test, y_train, y_test = train_test_split(X_transcript, calls_df['call_id'], test_size=0.3)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

# Predictions on test dataset
if 'call_id' in test_df.columns:
    X_test_transcript = vectorizer.transform(test_df['call_id'].fillna(''))  # Adjust according to test data
    predictions = clf.predict(X_test_transcript)
    output_df = pd.DataFrame({'call_id': test_df['call_id'], 'predicted_reason': predictions})
    output_df.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to test_predictions.csv")
