import streamlit as st
import pandas as pd
import plotly.express as px
from nytcomments.nytcomments import get_comments
from textblob import TextBlob

# Configure page
st.set_page_config(page_title="NYT Comments Analysis", layout="wide")
st.title("New York Times Comments Sentiment Analysis")

# Input section
article_url = st.text_input("Enter NYT Article URL:", 
                           placeholder="https://www.nytimes.com/...")

if article_url:
    # Fetch comments
    with st.spinner('Fetching comments from NYT...'):
        comments_df = get_comments([article_url])
        
    if not comments_df.empty:
        # Sentiment analysis
        comments_df['sentiment'] = comments_df['commentBody'].apply(
            lambda text: TextBlob(text).sentiment.polarity
        )
        
        # Convert to datetime and extract time
        comments_df['commentDate'] = pd.to_datetime(comments_df['createDate'])
        comments_df['time'] = comments_df['commentDate'].dt.time
        
        # Create size column (scale recommendations)
        max_rec = comments_df['recommendations'].max()
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
            title=f"Comment Sentiment Analysis: {article_url[:50]}..."
        )
        
        # Customize markers
        fig.update_traces(
            marker=dict(
                line=dict(width=1, color='DarkSlateGrey'),
                sizemode='diameter'
            ),
            selector=dict(mode='markers')
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
        st.warning("No comments found for this article.")
else:
    st.info("Please enter a valid NYT article URL to begin analysis")
