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

def extract_uuid(article_url):
    """Extract UUID from NYT article page metadata"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(article_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Method 1: JSON-LD metadata
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if data.get('@type') == 'NewsArticle':
                    url = data.get('mainEntityOfPage', {}).get('@id', '')
                    if match := re.search(r'article/([a-f0-9-]{36})', url):
                        return match.group(1)
            except:
                continue
        
        # Method 2: Open Graph URL
        meta_og = soup.find('meta', property='og:url')
        if meta_og and (match := re.search(r'article/([a-f0-9-]{36})', meta_og['content'])):
            return match.group(1)
        
        st.warning("UUID not found in page metadata")
        return None
        
    except Exception as e:
        st.error(f"Metadata extraction error: {str(e)}")
        return None

def get_comments_via_graphql(uuid):
    """Fetch comments using UUID with GraphQL API"""
    try:
        # GraphQL query parameters
        operation = "communityCommentsQuery"
        sha256_hash = "942211c7ff190af4945b8a45e7dfe97233a335a1537e74a1349515bc56c3518"
        
        # Construct GraphQL URL
        graphql_url = (
            f"https://samizdat-graphql.nytimes.com/graphql/v2?"
            f"operationName={operation}&"
            f"variables=%7B%22uri%22%3A%22nyt%3A%2F%2Farticle%2F{uuid}%22%2C%22input%22%3A%7B%22after%22%3Anull%2C%22asc%22%3Afalse%2C%22first%22%3A100%2C%22replies%22%3A3%2C%22view%22%3A%22ALL%22%7D%2C%22hasThreading%22%3Afalse%7D&"
            f"extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%22{sha256_hash}%22%7D%7D"
        )
        
        # Send request
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'referer': 'https://www.nytimes.com/'
        }
        response = requests.get(graphql_url, headers=headers)
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
    """Scrape comments directly from article page"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(article_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find comments script
        script_tag = soup.find('script', id='js-article-comments')
        if not script_tag:
            return pd.DataFrame()
        
        # Extract JSON data
        json_data = re.search(r'window\.__preloadedData\s*=\s*({.*?});', script_tag.string, re.DOTALL)
        if not json_data:
            return pd.DataFrame()
        
        data = json.loads(json_data.group(1))
        comments = data.get('comments', [])
        
        # Create DataFrame
        return pd.DataFrame(comments)[['commentBody', 'createDate', 'recommendations', 'editorsSelection', 'userDisplayName']]
    
    except Exception as e:
        st.error(f"Scraping Error: {str(e)}")
        return pd.DataFrame()

# Input section
article_url = st.text_input("Enter NYT Article URL:", 
                           placeholder="https://www.nytimes.com/...")

if article_url:
    # Extract UUID from page metadata
    uuid = extract_uuid(article_url)
    
    if uuid:
        st.success(f"Found UUID: {uuid}")
        with st.spinner('Fetching comments via GraphQL API...'):
            comments_df = get_comments_via_graphql(uuid)
    else:
        st.warning("Using HTML scraping method")
        with st.spinner('Fetching comments via HTML scraping...'):
            comments_df = scrape_nyt_comments(article_url)
    
    # Process and display results
    if not comments_df.empty:
        # Sentiment analysis
        comments_df['sentiment'] = comments_df['commentBody'].apply(
            lambda text: TextBlob(text).sentiment.polarity
        )
        
        # Convert to datetime
        comments_df['commentDate'] = pd.to_datetime(comments_df['createDate'])
        
        # Create size column
        max_rec = comments_df['recommendations'].max() or 1
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
            symbol='symbol',
            hover_name='userDisplayName',
            hover_data=['commentBody', 'recommendations', 'editorsSelection'],
            title="Comment Sentiment Analysis"
        )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data summary
        with st.expander("View Comment Data"):
            st.dataframe(comments_df[['commentDate', 'sentiment', 'recommendations', 'editorsSelection']])
    else:
        st.warning("No comments found for this article.")
else:
    st.info("Please enter a valid NYT article URL to begin analysis")
