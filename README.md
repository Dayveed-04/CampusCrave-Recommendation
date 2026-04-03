CampusCrave Recommendation Service
A lightweight content-based recommendation engine for CampusCrave — a food ordering system for tertiary institutions. Built with Python and Flask.

Overview
This service analyzes a student's order history and recommends food items they are likely to enjoy. It uses TF-IDF vectorization and cosine similarity to find menu items that are most similar to what the student has previously ordered.
How it works

Each menu item is represented as a feature string combining its category, name, description, and price range
Category is weighted more heavily (repeated 3x) to prioritize similar food types
The student's previously ordered items are averaged into a student profile vector
Cosine similarity is computed between the student profile and all available menu items
The top N most similar items are returned, with a maximum of 2 items per category to ensure variety

Price Range Bucketing
PriceLabelBelow ₦1,500budget₦1,500 – ₦2,500midrangeAbove ₦2,500premium

Tech Stack

Language: Python 3.x
Framework: Flask
ML Libraries: scikit-learn, NumPy
