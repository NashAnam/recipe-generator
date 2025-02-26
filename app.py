from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("C:/Users/Khan/Desktop/pics/recipes.csv")  # Replace with your actual dataset
df = df[['Recipe', 'Ingredients', 'Instructions', 'Image_URL']]  # Ensure dataset has an image column

# Convert ingredients into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Ingredients"])

# Function to recommend a recipe based on user input
def recommend_recipe(user_ingredients):
    user_input = [" ".join(user_ingredients)]
    user_vector = vectorizer.transform(user_input)
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    
    # Get top match
    top_match_index = similarity_scores.argsort()[0][-1]
    
    # Return recommended recipe details
    return df.iloc[top_match_index]

# Flask route to handle user requests
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_ingredients = request.form["ingredients"].lower().split(", ")
        recipe = recommend_recipe(user_ingredients)

        return render_template("index.html", 
                               recipe_name=recipe["Recipe"], 
                               instructions=recipe["Instructions"], 
                               image_url=recipe["Image_URL"])
    
    return render_template("index.html", recipe_name=None)

if __name__ == "__main__":
    app.run(debug=True)
