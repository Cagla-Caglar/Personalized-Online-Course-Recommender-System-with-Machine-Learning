import streamlit as st
st.set_page_config(page_title="🎓 Personalized Course Recommendation System", layout="wide")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from surprise import KNNBasic, NMF, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as st_train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_ratings():
    return pd.read_csv(os.path.join(DATA_DIR, 'course_ratings.csv'))

def load_courses():
    df = pd.read_csv(os.path.join(DATA_DIR, 'course_processed.csv'))
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_bow():
    return pd.read_csv(os.path.join(DATA_DIR, 'courses_bows.csv'))

def load_course_genres():
    return pd.read_csv(os.path.join(DATA_DIR, 'course_genre.csv'))

def load_user_profiles():
    return pd.read_csv(os.path.join(DATA_DIR, 'user_profile.csv'))

ratings_df = load_ratings()
courses_df = load_courses()
bow_df = load_bow()
course_genre_df = load_course_genres()
user_profiles_df = load_user_profiles()

def train_knn_model(k=10, min_k=3, sim_name='cosine', user_based=False):
    reader_knn = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
    data_knn = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader_knn)
    trainset_knn, testset_knn = st_train_test_split(data_knn, test_size=0.3, random_state=42)
    sim_options = {'name': sim_name, 'user_based': user_based}
    knn_model = KNNBasic(k=k, min_k=min_k, sim_options=sim_options)
    knn_model.fit(trainset_knn)
    test_rmse = accuracy.rmse(knn_model.test(testset_knn))
    return knn_model, test_rmse

def train_nmf_model(n_factors=32, init_low=0.5, init_high=5.0, n_epochs=20):
    reader_nmf = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
    nmf_dataset = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader_nmf)
    trainset_nmf, testset_nmf = st_train_test_split(nmf_dataset, test_size=0.3, random_state=42)
    nmf_model = NMF(n_factors=n_factors, init_low=init_low, init_high=init_high, 
                   n_epochs=n_epochs, random_state=123, verbose=False)
    nmf_model.fit(trainset_nmf)
    test_rmse = accuracy.rmse(nmf_model.test(testset_nmf))
    return nmf_model, test_rmse

def process_dataset(raw_data):
    df = raw_data.copy()
    user_list = df["user"].unique().tolist()
    course_list = df["item"].unique().tolist()
    user_id2idx = {x: i for i, x in enumerate(user_list)}
    course_id2idx = {x: i for i, x in enumerate(course_list)}
    df["user"] = df["user"].map(user_id2idx)
    df["item"] = df["item"].map(course_id2idx)
    df["rating"] = df["rating"].astype(int)
    return df, user_list, course_list

def generate_train_test_datasets(df, scale=True):
    min_rating = df["rating"].min()
    max_rating = df["rating"].max()
    df = df.sample(frac=1, random_state=42)
    x = df[["user", "item"]].values
    y = df["rating"].apply(lambda r: (r - min_rating) / (max_rating - min_rating)).values if scale else df["rating"].values
    n = df.shape[0]
    train_end = int(0.8 * n)
    test_start = int(0.9 * n)
    return x[:train_end], x[train_end:test_start], x[test_start:], y[:train_end], y[train_end:test_start], y[test_start:]

class OptimizedRecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=64, hidden_layers=[128, 64, 32], dropout_rate=0.3, l2_reg=0.001):
        super(OptimizedRecommenderNet, self).__init__()
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_regularizer=keras.regularizers.l2(l2_reg))
        self.item_embedding = layers.Embedding(num_items, embedding_size, embeddings_regularizer=keras.regularizers.l2(l2_reg))
        
        self.hidden_layers = []
        self.batch_norms = []
        for units in hidden_layers:
            self.hidden_layers.append(layers.Dense(units, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)))
            self.batch_norms.append(layers.BatchNormalization())
            
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(1, activation="linear")
        
    def call(self, inputs):
        user_vec = self.user_embedding(inputs[:, 0])
        item_vec = self.item_embedding(inputs[:, 1])
        x = tf.concat([user_vec, item_vec], axis=-1)
        x = layers.Flatten()(x)
        
        for i, (hidden, bn) in enumerate(zip(self.hidden_layers, self.batch_norms)):
            x = hidden(x)
            x = bn(x)
            if i < len(self.hidden_layers) - 1:  # Apply dropout to all but the last layer
                x = self.dropout(x)
                
        return self.output_layer(x)

def train_nn_model(embedding_size=64, hidden_layers=[128, 64, 32], learning_rate=0.0003, 
                  dropout_rate=0.3, l2_reg=0.001, batch_size=32, epochs=5):
    encoded_data, user_list_nn, course_list_nn = process_dataset(ratings_df)
    x_train, x_val, x_test, y_train, y_val, y_test = generate_train_test_datasets(encoded_data)
    num_users_nn = len(user_list_nn)
    num_items_nn = len(course_list_nn)
    
    nn_model = OptimizedRecommenderNet(num_users_nn, num_items_nn, embedding_size, 
                                     hidden_layers, dropout_rate, l2_reg)
    nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    with st.spinner('Training the neural network model, please wait...'):
        history = nn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                             validation_data=(x_val, y_val), verbose=1)
        
    # Evaluate on test set
    test_loss, test_rmse = nn_model.evaluate(x_test, y_test, verbose=0)
    
    return nn_model, user_list_nn, course_list_nn, test_rmse, history

def train_kmeans_model(n_clusters=8, max_iter=200, tol=1e-4, n_components=9):
    user_features = user_profiles_df.drop(columns=['user'])
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_features)
    
    pca = PCA(n_components=n_components, random_state=123)
    user_features_pca = pca.fit_transform(user_features_scaled)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=123, n_init=10, tol=tol, max_iter=max_iter)
    kmeans.fit(user_features_pca)
    
    # Calculate silhouette score as performance metric
    from sklearn.metrics import silhouette_score
    if n_clusters > 1 and len(user_features_pca) > n_clusters:
        silhouette = silhouette_score(user_features_pca, kmeans.labels_)
    else:
        silhouette = 0
    
    cluster_labels = kmeans.labels_
    cluster_df = pd.DataFrame({'user': user_profiles_df['user'], 'cluster': cluster_labels})
    
    return kmeans, cluster_df, silhouette, pca

def get_doc_dicts(bow_df):
    grouped = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    return idx_id_dict, id_idx_dict

def compute_similarity_matrix(bow_df):
    """Computes cosine similarity matrix from bag of words data"""

    idx_id_dict, id_idx_dict = get_doc_dicts(bow_df)
    valid_indices = list(idx_id_dict.keys())
    
    if not valid_indices:
        raise ValueError("Error: No matching indices found!")

    num_docs = len(valid_indices)
    word_col = 'token'  
    
    le = LabelEncoder()
    bow_df[word_col] = le.fit_transform(bow_df[word_col])  

    dtm = np.zeros((num_docs, bow_df[word_col].max() + 1))
    
    for _, row in bow_df.iterrows():
        doc_idx = row['doc_index']
        
        if doc_idx not in valid_indices:
            continue  
        
        word_idx = row[word_col]
        tf = row['bow']  
        dtm[doc_idx, word_idx] = tf

    if np.all(dtm == 0):
        raise ValueError("Error: Document-term matrix (dtm) is entirely zero!")

    sim_matrix = cosine_similarity(dtm)
    
    return sim_matrix, idx_id_dict, id_idx_dict

   
def generate_recommendations_for_one_user(enrolled_ids, unselected_ids, id_idx_dict, sim_matrix, threshold=0.6):
    res = {}
    for enrolled in enrolled_ids:
        for unselected in unselected_ids:
            if enrolled in id_idx_dict and unselected in id_idx_dict:
                sim = sim_matrix[id_idx_dict[enrolled], id_idx_dict[unselected]]
                if sim > threshold:
                    if unselected not in res or sim > res[unselected]:
                        res[unselected] = sim
    return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

def find_similar_users(selected_courses, num_similar=5):
    if not selected_courses:
        return []
    
    users_with_courses = ratings_df[ratings_df['item'].isin(selected_courses)]['user'].unique()
    
    if len(users_with_courses) == 0:
        return []
    
    return users_with_courses[:min(num_similar, len(users_with_courses))].tolist()

def recommend_courses_knn(knn_model, selected_courses, num_recommendations=10):
    similar_users = find_similar_users(selected_courses)
    
    if not similar_users:
        return pd.DataFrame(columns=["COURSE_ID", "TITLE"])
    
    similar_user_courses = ratings_df[ratings_df['user'].isin(similar_users)]
    recommended_courses = similar_user_courses['item'].value_counts().head(num_recommendations + len(selected_courses)).index.tolist()
    
    recommended_courses = [c for c in recommended_courses if c not in selected_courses][:num_recommendations]
    
    return courses_df[courses_df['COURSE_ID'].isin(recommended_courses)][["COURSE_ID", "TITLE"]]

def recommend_courses_nmf(nmf_model, selected_courses, num_recommendations=10):
    similar_users = find_similar_users(selected_courses)
    
    if not similar_users:
        return pd.DataFrame(columns=["COURSE_ID", "TITLE"])
    
    user_id = similar_users[0]
    
    preds = [nmf_model.predict(user_id, i).est for i in range(len(courses_df))]
    indices = np.argsort(preds)[-num_recommendations-len(selected_courses):][::-1]
    
    rec_courses = []
    for idx in indices:
        course_id = courses_df.iloc[idx]['COURSE_ID']
        if course_id not in selected_courses:
            rec_courses.append(course_id)
        if len(rec_courses) >= num_recommendations:
            break
    
    return courses_df[courses_df['COURSE_ID'].isin(rec_courses)][["COURSE_ID", "TITLE"]]

def recommend_courses_nn(nn_model, user_list_nn, course_list_nn, selected_courses, num_recommendations=10):
    similar_users = find_similar_users(selected_courses)
    
    if not similar_users:
        return pd.DataFrame(columns=["COURSE_ID", "TITLE"])
    
    user_id = similar_users[0]
    
    if user_id not in user_list_nn:
        return pd.DataFrame(columns=["COURSE_ID", "TITLE"])
        
    user_idx = user_list_nn.index(user_id)
    
    num_items_nn = len(course_list_nn)
    user_arr = np.full((num_items_nn, 1), user_idx)
    item_arr = np.arange(num_items_nn).reshape(-1, 1)
    x_pred = np.hstack([user_arr, item_arr])
    preds = nn_model.predict(x_pred, verbose=0).flatten()
    
    indices = np.argsort(preds)[-num_recommendations-len(selected_courses):][::-1]
    
    rec_ids = []
    for idx in indices:
        course_id = course_list_nn[idx]
        if course_id not in selected_courses:
            rec_ids.append(course_id)
        if len(rec_ids) >= num_recommendations:
            break
    
    return courses_df[courses_df['COURSE_ID'].isin(rec_ids)][["COURSE_ID", "TITLE"]]

def recommend_courses_kmeans(cluster_df, selected_courses, num_recommendations=10):
    similar_users = find_similar_users(selected_courses)
    
    if not similar_users:
        return pd.DataFrame(columns=["COURSE_ID", "TITLE"])
    
    similar_user_clusters = cluster_df[cluster_df['user'].isin(similar_users)]['cluster'].value_counts()
    if len(similar_user_clusters) == 0:
        return pd.DataFrame(columns=["COURSE_ID", "TITLE"])
    
    target_cluster = similar_user_clusters.index[0]
    
    cluster_users = cluster_df[cluster_df['cluster'] == target_cluster]['user'].tolist()
    
    cluster_ratings = ratings_df[ratings_df['user'].isin(cluster_users)]
    course_counts = cluster_ratings['item'].value_counts()
    
    course_counts = course_counts[~course_counts.index.isin(selected_courses)]
    recommended_courses = course_counts.head(num_recommendations).index.tolist()
    
    return courses_df[courses_df['COURSE_ID'].isin(recommended_courses)][["COURSE_ID", "TITLE"]]

def recommend_courses_similarity(selected_courses, bow_df, num_recommendations=10, threshold=0.6):
    # Compute similarity matrix on the fly
    sim_matrix, idx_id_dict, id_idx_dict = compute_similarity_matrix(bow_df)
    
    all_courses = set(courses_df['COURSE_ID'])
    unselected = list(all_courses.difference(set(selected_courses)))
    rec_dict = generate_recommendations_for_one_user(selected_courses, unselected, id_idx_dict, sim_matrix, threshold=threshold)
    rec_ids = list(rec_dict.keys())[:num_recommendations]
    
    # Create a dataframe with course IDs, titles, and similarity scores
    rec_df = courses_df[courses_df['COURSE_ID'].isin(rec_ids)][["COURSE_ID", "TITLE"]].copy()
    
    # Add the similarity scores
    rec_df["SIMILARITY_SCORE"] = rec_df["COURSE_ID"].apply(lambda x: round(rec_dict.get(x, 0) * 100, 2))
    
    return rec_df

def recommend_courses(model_type, selected_courses, num_recommendations, trained_models, bow_df=None):
    if not selected_courses:
        st.error("Please select at least one course to get recommendations.")
        return pd.DataFrame(columns=["COURSE_ID", "TITLE"])
    
    if model_type == "KNN":
        return recommend_courses_knn(trained_models["knn"], selected_courses, num_recommendations)
    elif model_type == "NMF":
        return recommend_courses_nmf(trained_models["nmf"], selected_courses, num_recommendations)
    elif model_type == "Neural Network":
        return recommend_courses_nn(trained_models["nn"], trained_models["user_list_nn"], 
                                   trained_models["course_list_nn"], selected_courses, num_recommendations)
    elif model_type == "Similarity":
        return recommend_courses_similarity(selected_courses, bow_df, num_recommendations, 
                                          trained_models["similarity_threshold"])
    elif model_type == "KMeans":
        return recommend_courses_kmeans(trained_models["cluster_df"], selected_courses, num_recommendations)
    else:
        return pd.DataFrame(columns=["COURSE_ID", "TITLE"])

# UI Section
st.title("🎓 Course Recommendation System")
st.markdown("Personalized course recommendations based on collaborative filtering, neural network embeddings, and content-based similarity.")

with st.sidebar:
    st.header("Configuration")
    model_type = st.selectbox("Select Recommendation Model", 
                             ["KNN", "NMF", "Neural Network", "Similarity", "KMeans"])
    
    num_recommendations = st.slider("Number of Recommendations", 1, 20, 10)
    
    st.header("Model Hyperparameters")
    show_advanced = st.checkbox("Show Advanced Hyperparameter Settings", False)
    
    model_params = {}
    performance_metrics = {}
    
    if show_advanced:
        if model_type == "KNN":
            st.subheader("KNN Hyperparameters")
            k = st.slider("Number of Neighbors (k)", 2, 30, 10)
            min_k = st.slider("Minimum Number of Neighbors (min_k)", 1, 10, 3)
            sim_options = st.selectbox("Similarity Metric", ["cosine", "msd", "pearson"], 0)
            user_based = st.checkbox("User-Based (vs Item-Based)", False)
            
            model_params["knn"] = {
                "k": k,
                "min_k": min_k,
                "sim_name": sim_options,
                "user_based": user_based
            }
            
        elif model_type == "NMF":
            st.subheader("NMF Hyperparameters")
            n_factors = st.slider("Number of Factors", 8, 64, 32)
            init_low = st.slider("Initial Low Value", 0.0, 1.0, 0.5)
            init_high = st.slider("Initial High Value", 1.0, 10.0, 5.0)
            n_epochs = st.slider("Number of Epochs", 5, 50, 20)
            
            model_params["nmf"] = {
                "n_factors": n_factors,
                "init_low": init_low,
                "init_high": init_high,
                "n_epochs": n_epochs
            }
            
        elif model_type == "Neural Network":
            st.subheader("Neural Network Hyperparameters")
            embedding_size = st.slider("Embedding Size", 16, 128, 64)
            dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.3)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.0003, format="%.4f")
            l2_reg = st.number_input("L2 Regularization", 0.0001, 0.01, 0.001, format="%.4f")
            batch_size = st.slider("Batch Size", 16, 128, 32, 8)
            epochs = st.slider("Number of Epochs", 2, 20, 5)
            
            hidden_layers = []
            st.write("Hidden Layers Configuration")
            layer1 = st.slider("Layer 1 Units", 32, 256, 128, 16)
            layer2 = st.slider("Layer 2 Units", 16, 128, 64, 8)
            layer3 = st.slider("Layer 3 Units", 8, 64, 32, 8)
            hidden_layers = [layer1, layer2, layer3]
            
            model_params["nn"] = {
                "embedding_size": embedding_size,
                "hidden_layers": hidden_layers,
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "l2_reg": l2_reg,
                "batch_size": batch_size,
                "epochs": epochs
            }
            
        elif model_type == "Similarity":
            st.subheader("Content-Based Similarity Hyperparameters")
            threshold = st.slider("Similarity Threshold", 0.3, 0.9, 0.6, 0.05)
            
            model_params["similarity"] = {
                "threshold": threshold
            }
            
        elif model_type == "KMeans":
            st.subheader("KMeans Hyperparameters")
            n_clusters = st.slider("Number of Clusters", 2, 15, 8)
            max_iter = st.slider("Maximum Iterations", 100, 500, 200, 50)
            tol = st.number_input("Convergence Tolerance", 1e-5, 1e-2, 1e-4, format="%.5f")
            n_components = st.slider("PCA Components", 2, 15, 9)
            
            model_params["kmeans"] = {
                "n_clusters": n_clusters,
                "max_iter": max_iter,
                "tol": tol,
                "n_components": n_components
            }
    else:
        # Default parameters for each model
        model_params["knn"] = {"k": 10, "min_k": 3, "sim_name": "cosine", "user_based": False}
        model_params["nmf"] = {"n_factors": 32, "init_low": 0.5, "init_high": 5.0, "n_epochs": 20}
        model_params["nn"] = {
            "embedding_size": 64, 
            "hidden_layers": [128, 64, 32],
            "learning_rate": 0.0003,
            "dropout_rate": 0.3,
            "l2_reg": 0.001,
            "batch_size": 32,
            "epochs": 5
        }
        model_params["similarity"] = {"threshold": 0.6}
        model_params["kmeans"] = {"n_clusters": 8, "max_iter": 200, "tol": 1e-4, "n_components": 9}

# Course selection
enrolled_courses_input = st.multiselect(
    "📌 Please select the courses you have completed or are interested in:", 
    options=courses_df['COURSE_ID'].tolist(),
    format_func=lambda x: courses_df.loc[courses_df['COURSE_ID'] == x, 'TITLE'].values[0]
)

if st.sidebar.button("🚀 Train Model & Generate Recommendations"):
    # Train the selected model with the chosen parameters
    trained_models = {}
    
    with st.spinner(f"Training {model_type} model with the selected parameters..."):
        if model_type == "KNN":
            knn_model, test_rmse = train_knn_model(**model_params["knn"])
            trained_models["knn"] = knn_model
            performance_metrics["rmse"] = test_rmse
            
        elif model_type == "NMF":
            nmf_model, test_rmse = train_nmf_model(**model_params["nmf"])
            trained_models["nmf"] = nmf_model
            performance_metrics["rmse"] = test_rmse
            
        elif model_type == "Neural Network":
            nn_model, user_list_nn, course_list_nn, test_rmse, history = train_nn_model(**model_params["nn"])
            trained_models["nn"] = nn_model
            trained_models["user_list_nn"] = user_list_nn
            trained_models["course_list_nn"] = course_list_nn
            performance_metrics["rmse"] = test_rmse
            performance_metrics["history"] = history
            
        elif model_type == "Similarity":
            trained_models["similarity_threshold"] = model_params["similarity"]["threshold"]
            performance_metrics["threshold"] = model_params["similarity"]["threshold"]
            
        elif model_type == "KMeans":
            kmeans_model, cluster_df, silhouette, pca = train_kmeans_model(**model_params["kmeans"])
            trained_models["kmeans"] = kmeans_model
            trained_models["cluster_df"] = cluster_df
            trained_models["pca"] = pca
            performance_metrics["silhouette"] = silhouette
    
    # Display model performance metrics
    st.sidebar.header("Model Performance")
    if model_type == "KNN" or model_type == "NMF":
        st.sidebar.metric("Test RMSE", f"{performance_metrics['rmse']:.4f}")
    elif model_type == "Neural Network":
        st.sidebar.metric("Test RMSE", f"{performance_metrics['rmse']:.4f}")
        
        # Plot training history
        if "history" in performance_metrics:
            history = performance_metrics["history"]
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(train_loss, label='Training Loss')
            ax.plot(val_loss, label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            st.sidebar.pyplot(fig)
            
    elif model_type == "KMeans":
        st.sidebar.metric("Silhouette Score", f"{performance_metrics['silhouette']:.4f}")
    
    # Generate recommendations
    with st.spinner("Generating recommendations..."):
        if model_type == "Similarity":
            rec_df = recommend_courses(model_type, enrolled_courses_input, num_recommendations, trained_models, bow_df)
        else:
            rec_df = recommend_courses(model_type, enrolled_courses_input, num_recommendations, trained_models)
            
        if not rec_df.empty:
            st.write("### 📢 Recommended Courses for You:")
            st.write(rec_df)
        else:
            st.error("Unable to generate recommendations. Please select different courses or try another model.")