import random
import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="NLP Dashboard - Streamlit App",
    page_icon="❤️",
)

def main():
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        df = pd.read_csv('nlp/data/dataPrabowo_cleaned.csv')

        st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:30px;justify-content:center;text-align:center;color:#ffb6c1">
                <h1 style="font-size:3rem;color:#ffb6c1">NLP X Analysis of "Prabowo"</h1>
            </div>
        """, unsafe_allow_html=True)

        st.image("nlp/src/image1.jpg")

        col1, col2 = st.columns(2)

        # Kolom 1 - Keywords
        with col1:
            st.markdown(f"""
                <div style="display:flex;flex-direction:column;align-items:center; gap: 10px">
                    <text style="font-size:1.2rem;font-weight:bold;color:#ffb6c1;transform:translateY(30px);text-align:center">Keywords</text>
                    <text style="font-size:2rem;font-weight:bolder;text-align:center">"prabowo lang:id"</text>
                </div>
            """, unsafe_allow_html=True)

        # Kolom 2 - Data Found
        with col2:
            st.markdown(f"""
                <div style="display:flex;flex-direction:column;align-items:center; gap: 10px">
                    <text style="font-size:1.2rem;font-weight:bold;color:#ffb6c1;transform:translateY(30px);text-align:center">Data Found</text>
                    <text style="font-size:3.5rem;font-weight:bolder;text-align:center">{len(df)}</text>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div style="display:flex;flex-direction:column;border:1px;align-items:center">
                <text style="font-size:1.2rem;font-weight:bold;color:#ffb6c1">Word Cloud</text>
            </div>
        """, unsafe_allow_html=True)

        def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            colors = ['#ffb6c1', '#f3c9d7', '#ffdae5']
            return random.choice(colors)

        words = " ".join(df["cleaning_stemmed"])
        wordcloud = WordCloud(width=800, height=400, background_color="#faf7f0", color_func=custom_color_func).generate(words)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        st.pyplot(fig)

        st.markdown("""
            <div style="display:flex;flex-direction:column;border:1px;align-items:center;margin-top:20px">
                <text style="font-size:1.2rem;font-weight:bold;color:#ffb6c1">Top 10 Words</text>
            </div>
        """, unsafe_allow_html=True)

        word_freq = wordcloud.words_
        top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:10])

        top_words_df = pd.DataFrame(top_words.items(), columns=['Kata', 'Frekuensi'])

        fig = px.bar(top_words_df, x='Kata', y='Frekuensi',
                     color='Frekuensi',
                     color_continuous_scale=['#ffdae5', '#ff7f9d', '#ffb6c1', '#d64768'])

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
            <div style="display:flex;flex-direction:column;border:1px;align-items:center">
                <text style="font-size:1.2rem;font-weight:bold;color:#ffb6c1">Clustering</text>
            </div>
        """, unsafe_allow_html=True)

        image = Image.open("nlp/src/ss1.png")
        st.image(image)

    elif choice == "About":

        st.subheader("About")
        st.markdown("""
            This is an NLP Dashboard built with Streamlit for analyzing tweets related to Prabowo.
            The dashboard visualizes the most frequent words and performs basic sentiment analysis using word clouds and clustering.
        """)

if __name__ == "__main__":
    main()