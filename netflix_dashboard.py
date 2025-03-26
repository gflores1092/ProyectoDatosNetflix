# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Netflix Content Dashboard",
    page_icon="🎬",
    layout="wide"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        with st.spinner('Downloading required NLTK data...'):
            nltk.download('vader_lexicon', quiet=True)

# Download NLTK data
download_nltk_data()

# Add title and description
st.title("🎬 Netflix Content Analysis Dashboard")
st.markdown("""
This dashboard provides insights into Netflix's content library, including movies and TV shows.
Explore the data using filters and interactive visualizations below.
""")

# Load and clean the data
@st.cache_data
def load_data():
    # Read the CSV file
    df = pd.read_csv('netflix_titles.csv')
    
    # Clean column names (convert to lowercase and replace spaces with underscores)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Handle missing values
    df['country'] = df['country'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('No cast information')
    df['director'] = df['director'].fillna('No director information')
    df['rating'] = df['rating'].fillna('Not rated')
    
    # Convert date_added to datetime
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    
    return df

# Load the data
df = load_data()

# Create sidebar filters
st.sidebar.header("Filters")

# Country filter
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=sorted(df['country'].unique()),
    default=['United States', 'India', 'United Kingdom']
)

# Type filter (Movie/TV Show)
selected_types = st.sidebar.multiselect(
    "Select Content Type",
    options=sorted(df['type'].unique()),
    default=df['type'].unique()
)

# Actor filter
# Extract unique actors from the cast column
all_actors = []
for actors in df['cast'].str.split(', '):
    if actors != 'No cast information':
        all_actors.extend(actors)
unique_actors = sorted(set(all_actors))

selected_actors = st.sidebar.multiselect(
    "Select Actors",
    options=unique_actors,
    default=[]
)

# Director filter
# Extract unique directors from the director column
unique_directors = sorted(df['director'].unique())
unique_directors = [d for d in unique_directors if d != 'No director information']

selected_directors = st.sidebar.multiselect(
    "Select Directors",
    options=unique_directors,
    default=[]
)

# Year range filter
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['release_year'].min()),
    max_value=int(df['release_year'].max()),
    value=(int(df['release_year'].min()), int(df['release_year'].max()))
)

# Apply filters
filtered_df = df[
    (df['country'].isin(selected_countries)) &
    (df['type'].isin(selected_types)) &
    (df['release_year'].between(year_range[0], year_range[1]))
]

# Apply actor filter if any actors are selected
if selected_actors:
    filtered_df = filtered_df[
        filtered_df['cast'].apply(
            lambda x: any(actor in str(x) for actor in selected_actors)
        )
    ]

# Apply director filter if any directors are selected
if selected_directors:
    filtered_df = filtered_df[
        filtered_df['director'].isin(selected_directors)
    ]

# Create two columns for charts
col1, col2 = st.columns(2)

# Genre distribution chart
with col1:
    # Split the listed_in column and create a list of all genres
    all_genres = []
    for genres in filtered_df['listed_in'].str.split(', '):
        all_genres.extend(genres)
    
    # Count genres
    genre_counts = pd.Series(all_genres).value_counts().head(10)
    
    # Create bar chart
    fig_genres = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title="Top 10 Genres",
        labels={'x': 'Genre', 'y': 'Number of Titles'}
    )
    st.plotly_chart(fig_genres, use_container_width=True)

# Time series chart
with col2:
    # Group by release year and type, count occurrences
    content_by_year = filtered_df.groupby(['release_year', 'type']).size().reset_index(name='count')
    
    # Create stacked bar chart
    fig_time = px.bar(
        content_by_year,
        x='release_year',
        y='count',
        color='type',
        title="Content Distribution by Year",
        labels={
            'release_year': 'Release Year',
            'count': 'Number of Titles',
            'type': 'Content Type'
        },
        barmode='stack'
    )
    
    # Update layout for better readability
    fig_time.update_layout(
        xaxis_title="Release Year",
        yaxis_title="Number of Titles",
        legend_title="Content Type",
        showlegend=True
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

# Display filtered data table
st.header("Filtered Content")
st.dataframe(filtered_df)

# Top Actors Analysis
st.header("Top Actors Analysis")
col5, col6 = st.columns(2)

with col5:
    # Split cast column and create a list of all actors
    all_actors = []
    for actors in filtered_df['cast'].str.split(', '):
        if actors != 'No cast information':
            all_actors.extend(actors)
    
    # Count actors and get top 10
    actor_counts = pd.Series(all_actors).value_counts().head(10)
    
    # Create bar chart for top actors
    fig_actors = px.bar(
        x=actor_counts.index,
        y=actor_counts.values,
        title="Top 10 Actors by Number of Appearances",
        labels={'x': 'Actor', 'y': 'Number of Appearances'}
    )
    fig_actors.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_actors, use_container_width=True)

with col6:
    # Create word cloud from movie descriptions
    st.subheader("Movie Descriptions Word Cloud")
    
    # Combine all descriptions
    all_descriptions = ' '.join(filtered_df['description'].dropna())
    
    # Create and generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        max_font_size=150,
        random_state=42,
        colormap='viridis'
    ).generate(all_descriptions)
    
    # Display the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Add Sentiment Analysis Section
st.header("Sentiment Analysis of Content Descriptions")
col7, col8 = st.columns(2)

with col7:
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Function to get sentiment scores
    def get_sentiment_scores(text):
        if pd.isna(text):
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
        return sia.polarity_scores(text)
    
    # Calculate sentiment scores for each description
    filtered_df['sentiment_scores'] = filtered_df['description'].apply(get_sentiment_scores)
    
    # Extract compound scores
    filtered_df['sentiment'] = filtered_df['sentiment_scores'].apply(lambda x: x['compound'])
    
    # Calculate average sentiment by content type
    sentiment_by_type = filtered_df.groupby('type')['sentiment'].mean().reset_index()
    
    # Create bar chart for sentiment by content type
    fig_sentiment = px.bar(
        sentiment_by_type,
        x='type',
        y='sentiment',
        title="Average Sentiment by Content Type",
        labels={
            'type': 'Content Type',
            'sentiment': 'Sentiment Score'
        },
        color='sentiment',
        color_continuous_scale=['red', 'yellow', 'green']
    )
    
    # Update layout
    fig_sentiment.update_layout(
        showlegend=False,
        yaxis_title="Sentiment Score (-1 to 1)",
        yaxis=dict(range=[-1, 1])
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)

with col8:
    # Create sentiment distribution plot
    fig_sentiment_dist = px.histogram(
        filtered_df,
        x='sentiment',
        color='type',
        nbins=50,
        title="Sentiment Distribution by Content Type",
        labels={
            'sentiment': 'Sentiment Score',
            'count': 'Number of Titles'
        }
    )
    
    # Update layout
    fig_sentiment_dist.update_layout(
        xaxis=dict(range=[-1, 1]),
        xaxis_title="Sentiment Score (-1 to 1)",
        yaxis_title="Number of Titles"
    )
    
    st.plotly_chart(fig_sentiment_dist, use_container_width=True)

# Add sentiment statistics
st.subheader("Sentiment Statistics")
col9, col10, col11 = st.columns(3)

with col9:
    # Calculate percentage of positive content
    positive_content = filtered_df[filtered_df['sentiment'] > 0.2].shape[0]
    total_content = filtered_df.shape[0]
    positive_percentage = (positive_content / total_content) * 100
    st.metric("Positive Content", f"{positive_percentage:.1f}%")

with col10:
    # Calculate percentage of neutral content
    neutral_content = filtered_df[
        (filtered_df['sentiment'] >= -0.2) & 
        (filtered_df['sentiment'] <= 0.2)
    ].shape[0]
    neutral_percentage = (neutral_content / total_content) * 100
    st.metric("Neutral Content", f"{neutral_percentage:.1f}%")

with col11:
    # Calculate percentage of negative content
    negative_content = filtered_df[filtered_df['sentiment'] < -0.2].shape[0]
    negative_percentage = (negative_content / total_content) * 100
    st.metric("Negative Content", f"{negative_percentage:.1f}%")

# Bonus: Content Duration Analysis
st.header("Content Duration Analysis")
col3, col4 = st.columns(2)

with col3:
    # Extract numeric duration values
    def extract_duration(duration_str):
        try:
            # Split by space and take the first part (number)
            return float(duration_str.split()[0])
        except:
            return None
    
    # Create a copy of filtered_df to avoid modifying the original
    duration_df = filtered_df.copy()
    duration_df['duration_numeric'] = duration_df['duration'].apply(extract_duration)
    
    # Calculate average duration by type
    duration_by_type = duration_df.groupby('type')['duration_numeric'].mean().reset_index()
    
    # Create bar chart for average duration
    fig_duration = px.bar(
        duration_by_type,
        x='type',
        y='duration_numeric',
        title="Average Duration by Content Type",
        labels={
            'type': 'Content Type',
            'duration_numeric': 'Average Duration (minutes/seasons)'
        }
    )
    st.plotly_chart(fig_duration, use_container_width=True)

with col4:
    # Create a scatter plot of duration vs. release year
    fig_scatter = px.scatter(
        duration_df,
        x='release_year',
        y='duration_numeric',
        color='type',
        title="Content Duration vs. Release Year",
        labels={
            'release_year': 'Release Year',
            'duration_numeric': 'Duration (minutes/seasons)'
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Add density plot for duration distribution
st.subheader("Duration Distribution by Content Type")
fig_density = px.histogram(
    duration_df,
    x='duration_numeric',
    color='type',
    nbins=50,
    marginal='box',
    title="Duration Distribution by Content Type",
    labels={
        'duration_numeric': 'Duration (minutes/seasons)',
        'count': 'Number of Titles'
    }
)
fig_density.update_layout(
    showlegend=True,
    legend_title="Content Type"
)
st.plotly_chart(fig_density, use_container_width=True) 