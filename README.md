# CampusCrave Recommendation Service

A content-based recommendation service for CampusCrave that suggests meals based on a student's previous orders.

The recommendation engine uses **TF-IDF vectorization** and **cosine similarity** to identify menu items that closely match a student's preferences.

---

## Features

- Personalized meal recommendations
- Content-based filtering
- TF-IDF vectorization
- Cosine similarity matching
- Price range awareness
- Category weighting for better relevance
- REST API built with Flask

---

## How It Works

1. Menu items are converted into feature vectors using:
   - Category
   - Food name
   - Description
   - Price range

2. Categories are weighted more heavily to prioritize similar food types.

3. A user profile is created from the student's previous orders.

4. Cosine similarity is calculated between the user profile and all available menu items.

5. The most relevant meals are returned while limiting recommendations from the same category to improve variety.

---

## Price Categories

| Price | Label |
|--------|-------|
| Below ₦1,500 | Budget |
| ₦1,500 – ₦2,500 | Mid-range |
| Above ₦2,500 | Premium |

---

## Tech Stack

- Python
- Flask
- scikit-learn
- NumPy
- Pandas

---

## Getting Started

```bash
git clone https://github.com/Dayveed-04/CampusCrave-Recommendation.git

cd CampusCrave-Recommendation

pip install -r requirements.txt

python app.py
```

The service runs on:

```
http://localhost:5001
```

---

## Future Improvements

- Hybrid recommendation model
- Better handling of new users (cold start)
- User feedback integration
- Improved ranking algorithm

---

## Author

David Uwaje

GitHub: https://github.com/Dayveed-04

LinkedIn: https://www.linkedin.com/in/david-uwaje-58153425b/
