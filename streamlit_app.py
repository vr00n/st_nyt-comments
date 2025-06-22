import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import requests
import json
import re
from bs4 import BeautifulSoup

# Configure page
st.set_page_config(page_title="NYT Comments Analysis", layout="wide")
st.title("New York Times Comments Sentiment Analysis")

def get_article_id(article_url):
    """Extract article ID from URL with detailed debugging"""
    st.info(f"Extracting ID from: {article_url}")
    patterns = [
        r'article/([a-f0-9-]+)',          # UUID pattern
        r'/(\d{4}/\d{2}/\d{2}/[^/]+)',    # Date/slug pattern
        r'(\d{5,})'                        # Numeric ID pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, article_url)
        if match:
            article_id = match.group(1)
            st.success(f"Pattern matched: {pattern}\nExtracted ID: {article_id}")
            return article_id
    
    st.warning("No matching pattern found in URL")
    return None

def get_comments_via_graphql(article_id):
    """Fetch comments using NYT's GraphQL API"""
    try:
        st.info(f"Fetching comments via GraphQL for ID: {article_id}")
        # GraphQL query parameters
        operation = "communityCommentsQuery"
        sha256_hash = "942211c7ff190af4945b8a45e7dfe97233a335a1537e74a1349515bc56c3518"
        uri = f"nyt://article/{article_id}"
        
        # Construct GraphQL URL
        graphql_url = (
            f"https://samizdat-graphql.nytimes.com/graphql/v2?"
            f"operationName={operation}&"
            f"variables=%7B%22uri%22%3A%22{uri}%22%2C%22input%22%3A%7B%22after%22%3Anull%2C%22asc%22%3Afalse%2C%22first%22%3A100%2C%22replies%22%3A3%2C%22view%22%3A%22ALL%22%7D%2C%22hasThreading%22%3Afalse%7D&"
            f"extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%22{sha256_hash}%22%7D%7D"
        )
        
        # Send request with headers
        headers = {
            'accept': '*/*',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'nyt-app-type': 'project-vi',
            'nyt-app-version': '0.0.5',
            'referer': 'https://www.nytimes.com/'
        }
        response = requests.get(graphql_url, headers=headers)
        
        # Debug API response
        with st.expander("GraphQL API Response"):
            st.json(response.json())
        
        data = response.json()
        
        # Process comments
        comments = []
        for edge in data['data']['communityComments']['edges']:
            comment = edge['node']['comment']
            comments.append({
                'commentBody': comment['text'],
                'createDate': comment['acceptedAt'],
                'recommendations': comment['recommendedCount'],
                'editorsSelection': comment['timesPick'],
                'userDisplayName': comment['author']['name']
            })
            
        return pd.DataFrame(comments)
    
    except Exception as e:
        st.error(f"GraphQL Error: {str(e)}")
        return pd.DataFrame()

def scrape_nyt_comments(article_url):
    """Fallback: Scrape comments directly from article page"""
    try:
        st.info("Falling back to HTML scraping method")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(article_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Debug HTML elements
        with st.expander("Page Content Analysis"):
            st.write(f"Title: {soup.title.string if soup.title else 'No title found'}")
            st.write(f"Script tags found: {len(soup.find_all('script'))}")
        
        # Find the script tag containing comment data
        script_tag = soup.find('script', id='js-article-comments')
        if not script_tag:
            st.warning("No comments script tag found in HTML")
            return pd.DataFrame()
        
        # Extract JSON data using regex
        json_data = re.search(r'window\.__preloadedData\s*=\s*({.*?});', script_tag.string, re.DOTALL)
        if not json_data:
            st.warning("No JSON data found in script tag")
            return pd.DataFrame()
        
        try:
            data = json.loads(json_data.group(1))
            comments = data.get('comments', [])
            
            # Create DataFrame with required columns
            comments_df = pd.DataFrame(comments)
            
            # Rename columns to match structure
            column_map = {
                'commentBody': 'commentBody',
                'createDate': 'createDate',
                'recommendations': 'recommendations',
                'editorsSelection': 'editorsSelection',
                'userDisplayName': 'userDisplayName'
            }
            return comments_df.rename(columns=column_map)[list(column_map.values())]
        
        except json.JSONDecodeError:
            st.error("Failed to parse JSON data")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Scraping Error: {str(e)}")
        return pd.DataFrame()

# Input section
article_url = st.text_input("Enter NYT Article URL:", 
                           placeholder="https://www.nytimes.com/...")

if article_url:
    # Debugging panel
    with st.expander("Debug Information"):
        st.subheader("URL Analysis")
        st.write(f"Input URL: `{article_url}`")
        
        # Extract article ID
        article_id = get_article_id(article_url)
        
        if article_id:
            # Try GraphQL method first
            with st.spinner('Fetching comments via GraphQL API...'):
                comments_df = get_comments_via_graphql(article_id)
            
            if comments_df.empty:
                st.warning("GraphQL method returned no comments. Falling back to HTML scraping.")
                with st.spinner('Fetching comments via HTML scraping...'):
                    comments_df = scrape_nyt_comments(article_url)
        else:
            st.warning("Using HTML scraping method directly")
            with st.spinner('Fetching comments via HTML scraping...'):
                comments_df = scrape_nyt_comments(article_url)
        
        st.subheader("Raw Comment Data Preview")
        st.write(comments_df.head(3) if not comments_df.empty else "No comments found")
    
    # Process and display comments if found
    if not comments_df.empty:
        # Sentiment analysis
        comments_df['sentiment'] = comments_df['commentBody'].apply(
            lambda text: TextBlob(text).sentiment.polarity
        )
        
        # Convert to datetime
        comments_df['commentDate'] = pd.to_datetime(comments_df['createDate'])
        
        # Create size column (scale recommendations)
        max_rec = comments_df['recommendations'].max() or 1  # Avoid division by zero
        comments_df['size'] = 10 + 40 * (comments_df['recommendations'] / max_rec)
        
        # Create marker symbols
        comments_df['symbol'] = comments_df['editorsSelection'].map({
            True: 'star',
            False: 'circle'
        })
        
        # Create plot
        fig = px.scatter(
            comments_df,
            x='commentDate',
            y='sentiment',
            size='size',
            color='recommendations',
            color_continuous_scale='Viridis',
            symbol='symbol',
            hover_name='userDisplayName',
            hover_data=['commentBody', 'recommendations', 'editorsSelection'],
            labels={
                'commentDate': 'Time',
                'sentiment': 'Sentiment Polarity',
                'recommendations': 'Likes'
            },
            title="Comment Sentiment Analysis"
        )
        
        # Customize layout
        fig.update_layout(
            hovermode='closest',
            xaxis_title='Comment Time',
            yaxis_title='Sentiment (-1 to 1)',
            coloraxis_colorbar=dict(title='Likes'),
            legend_title_text='Marker Meaning'
        )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data summary
        with st.expander("View Comment Data"):
            st.dataframe(comments_df[['commentDate', 'sentiment', 
                                     'recommendations', 'editorsSelection']])
    else:
        st.warning("No comments found for this article after all methods")
else:
    st.info("Please enter a valid NYT article URL to begin analysis")
