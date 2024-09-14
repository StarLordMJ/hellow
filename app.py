import streamlit as st
import pickle

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load the model
loaded_model = load_model('recommendation_model.pkl')

def get_recommendations(input_categories):
    tfidf_vectorizer = loaded_model['tfidf_vectorizer']
    knn_classifier = loaded_model['knn_classifier']
    df = loaded_model['df']

    all_predictions = []

    for category in input_categories:
        category_tfidf = tfidf_vectorizer.transform([category])
        top_20_indices = knn_classifier.kneighbors(category_tfidf, n_neighbors=20, return_distance=False)[0]
        top_20_places = df.iloc[top_20_indices]
        best_2_places = top_20_places.sort_values('normalized_score', ascending=False).head(2)
        
        all_predictions.append({
            'category': category,
            'predictions': best_2_places[['name', 'rating', 'user_ratings_total', 'normalized_score']].to_dict('records')
        })

    return all_predictions

# Streamlit app
st.title("Place Recommendation System")

st.write("Enter the categories you are interested in (comma-separated):")
input_categories = st.text_input("Categories", "historical monuments, history tours, wildlife")

if st.button("Get Recommendations"):
    categories_list = [category.strip() for category in input_categories.split(',')]
    recommendations = get_recommendations(categories_list)

    for prediction in recommendations:
        st.subheader(f"Top 2 recommended places for '{prediction['category']}':")
        for place in prediction['predictions']:
            st.write(f"**Name:** {place['name']}")
            st.write(f"**Rating:** {place['rating']:.2f}")
            st.write(f"**Review Count:** {place['user_ratings_total']}")
            st.write(f"**Score:** {place['normalized_score']:.4f}")
            st.write("")  # Add a blank line for better spacing