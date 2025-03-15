Personalized Course Recommendation System

This project is a machine learning-based Course Recommendation System developed as part of the IBM Machine Learning Capstone. While IBM's structured learning approach guided the analysis and feature engineering, the recommendation system and Streamlit-based application were independently developed. Key model parameters and insights derived from these analyses were utilized to optimize the recommendation process.

Key Features

Interactive Model Selection & Training:
Users can choose between multiple recommendation models and configure hyperparameters in real time. The selected model is trained instantly based on user-defined parameters, providing immediate feedback on performance metrics.

Content-Based Filtering:
Utilizes course descriptions and keywords to compute similarity scores using Bag-of-Words (BoW) and cosine similarity.

Collaborative Filtering:
Implements user-item interaction-based recommendations using K-Nearest Neighbors (KNN), Non-Negative Matrix Factorization (NMF), and Deep Learning-based User-Item Interaction Modeling with Neural Network Embeddings.

Deep Learning-Based Recommendations:
A neural network embedding model is trained to capture latent relationships between users and courses based on their interaction patterns for enhanced personalization.

Clustering-Based Recommendations:
Groups users with similar learning preferences using K-Means clustering on PCA-transformed feature sets, allowing recommendations tailored to segmented user profiles.

Real-Time Performance Evaluation:
Users can modify hyperparameters via the sidebar and observe how these changes impact the model’s performance through interactive metrics such as RMSE and silhouette scores.

Scalable Design:
Efficiently processes large datasets, making it suitable for real-world educational platforms.

Web Application
The interactive web application is deployed using Streamlit and allows users to:

- Select completed courses to personalize recommendations.
- Choose a recommendation model from KNN, NMF, Deep Learning-based Collaborative Filtering, K-Means Clustering, and Content-Based Filtering using Cosine Similarity.
- Adjust hyperparameters (e.g., number of neighbors in KNN, embedding size in Neural Networks, number of clusters in K-Means).
- Train the model in real time and receive immediate recommendations.
- Evaluate model performance through dynamic metrics such as RMSE and silhouette scores.

📌 Access the Web Application:
👉 https://personalized-online-course-recommender-system.streamlit.app/

Repository Contents

This repository includes:

- Python scripts for model development, training, and evaluation.
- Preprocessed datasets used for recommendation system training and validation.
- Implementation of multiple recommendation techniques, including K-Nearest Neighbors (KNN), Non-Negative Matrix Factorization (NMF), Deep Learning-based Collaborative Filtering using Neural Network Embeddings, Content-Based Filtering using Cosine Similarity, and K-Means Clustering.
- Streamlit application code, enabling a fully interactive user experience.

The repository also includes a detailed presentation summarizing the entire analysis, methodologies, and future recommendations. This document serves as the project's final report, providing an in-depth overview of the recommendation system's design and performance evaluation.

License

This project is licensed under the MIT License.

Developed by Çağla Çağlar as part of the IBM Machine Learning Capstone Course.

