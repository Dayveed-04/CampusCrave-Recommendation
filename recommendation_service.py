
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)


def build_menu_features(menu):
    """
    Combine menu features into a single string for TF-IDF vectorization.
    Category name is repeated to give it more weight.
    """
    category = menu.get("category", {}).get("name", "")
    name = menu.get("name", "")
    description = menu.get("description", "")

    # Price range bucketing
    price = menu.get("basePrice", 0)
    if price < 1500:
        price_range = "budget"
    elif price < 2500:
        price_range = "midrange"
    else:
        price_range = "premium"

    # Repeat category 3x to give it more weight in similarity
    feature_string = f"{category} {category} {category} {name} {description} {price_range}"
    return feature_string.lower()


def get_recommendations(menus, ordered_menu_ids, top_n=5):
    if not menus or not ordered_menu_ids:
        return [m["id"] for m in menus[:top_n]]

    menu_ids = [m["id"] for m in menus]
    feature_strings = [build_menu_features(m) for m in menus]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(feature_strings)

    ordered_indices = [
        menu_ids.index(mid) for mid in ordered_menu_ids if mid in menu_ids
    ]

    if not ordered_indices:
        return [m["id"] for m in menus[:top_n]]

    ordered_vectors = tfidf_matrix[ordered_indices]
    student_profile = np.asarray(ordered_vectors.mean(axis=0))

    similarities = cosine_similarity(student_profile, tfidf_matrix).flatten()
    sorted_indices = np.argsort(similarities)[::-1]

    recommendations = []
    category_counts = {}  # ← track how many per category
    max_per_category = 2  # ← max 2 items per category

    for idx in sorted_indices:
        menu_id = menu_ids[idx]
        if menu_id in ordered_menu_ids:
            continue

        # Get category of this menu
        menu = menus[idx]
        category = menu.get("category", {}).get("name", "unknown")

        # Skip if this category already has max items
        if category_counts.get(category, 0) >= max_per_category:
            continue

        recommendations.append(menu_id)
        category_counts[category] = category_counts.get(category, 0) + 1

        if len(recommendations) >= top_n:
            break

    return recommendations


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /recommend
    Body:
    {
        "menus": [...],        # all available menus from your API
        "ordered_menu_ids": [1, 4, 5, ...],  # menu IDs this student has ordered
        "top_n": 5             # optional, default 5
    }

    Returns:
    {
        "status": "success",
        "recommended_menu_ids": [3, 6, 7, ...]
    }
    """
    try:
        data = request.get_json()

        menus = data.get("menus", [])
        ordered_menu_ids = data.get("ordered_menu_ids", [])
        top_n = data.get("top_n", 5)

        if not menus:
            return jsonify({"status": "error", "message": "No menus provided"}), 400

        recommended_ids = get_recommendations(menus, ordered_menu_ids, top_n)

        return jsonify({
            "status": "success",
            "recommended_menu_ids": recommended_ids
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "food-recommendation"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)