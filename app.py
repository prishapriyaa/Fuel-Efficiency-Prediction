# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import hashlib
from sklearn.metrics import root_mean_squared_error

# Set up Streamlit page
st.set_page_config(page_title="Fuel Efficiency Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("final_data.csv")
    df.columns = df.columns.str.lower().str.strip()

    # Clean numerical columns
    df['max_power'] = df['max_power'].str.replace(' bhp', '', regex=False).astype(float)
    df['mileage'] = df['mileage'].str.replace(' kmpl', '', regex=False).str.replace(' km/kg', '', regex=False).astype(float)

    df = df.dropna(subset=['max_power', 'mileage', 'engine', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'])

    # Encode categorical variables
    fuel_enc = LabelEncoder()
    seller_enc = LabelEncoder()
    trans_enc = LabelEncoder()
    owner_enc = LabelEncoder()

    df['fuel_enc'] = fuel_enc.fit_transform(df['fuel'])
    df['seller_enc'] = seller_enc.fit_transform(df['seller_type'])
    df['trans_enc'] = trans_enc.fit_transform(df['transmission'])
    df['owner_enc'] = owner_enc.fit_transform(df['owner'])

    return df, fuel_enc, seller_enc, trans_enc, owner_enc

def hash_graph(graph):
    # Convert important tensors to bytes and hash them
    m = hashlib.sha256()
    for attr in ['x', 'edge_index', 'edge_attr', 'y']:
        if hasattr(graph, attr) and getattr(graph, attr) is not None:
            tensor = getattr(graph, attr)
            m.update(tensor.cpu().numpy().tobytes())
    return m.hexdigest()

@st.cache_data
def preprocess_features(df):
    features = ['year', 'km_driven', 'engine', 'max_power', 'fuel_enc', 'seller_enc', 'trans_enc', 'owner_enc']
    target = 'mileage'
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, y, X_scaled, scaler

def split_graph_data(data, train_ratio=0.8):
    total = data.num_nodes
    indices = np.random.permutation(total)
    train_size = int(train_ratio * total)
    
    train_mask = torch.zeros(total, dtype=torch.bool)
    test_mask = torch.zeros(total, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True

    data.train_mask = train_mask
    data.test_mask = test_mask
    return data

@st.cache_resource
def train_tabnet(X_train, y_train, X_val, y_val):
    tabnet = TabNetRegressor()
    tabnet.fit(
        X_train=X_train, y_train=y_train.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        max_epochs=100,
        patience=10
    )
    y_pred = tabnet.predict(X_val).flatten()
    rmse = root_mean_squared_error(y_val, y_pred)
    return tabnet, mean_absolute_error(y_val, y_pred), r2_score(y_val, y_pred), rmse

class GNNRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super(GNNRegressor, self).__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.lin = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x

@st.cache_data
def build_knn_graph(X, y=None, k=10):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    knn_graph = nbrs.kneighbors_graph(X).toarray()

    edge_index = []
    for i in range(knn_graph.shape[0]):
        for j in range(knn_graph.shape[1]):
            if knn_graph[i][j] and i != j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(X, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index)

    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return data

@st.cache_resource
def train_gnn_whole_graph(graph_hash, _graph_data, epochs=600, lr=0.005):
    data = split_graph_data(_graph_data)  # Now data is defined properly

    model = GNNRegressor(input_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train the model
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(data)[data.test_mask].squeeze().numpy()
        true = data.y[data.test_mask].squeeze().numpy()

    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = root_mean_squared_error(true, pred)
    return model, mae, r2, rmse

def predict_gnn_live(model, full_X_scaled, full_y, scaler, user_input, k=5):
    user_scaled = scaler.transform(user_input)
    nbrs = NearestNeighbors(n_neighbors=k).fit(full_X_scaled)
    distances, indices = nbrs.kneighbors(user_scaled)

    edge_index = []
    for i, idx in enumerate(indices[0]):
        edge_index.append([0, i + 1])
        edge_index.append([i + 1, 0])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    neighbor_features = full_X_scaled[indices[0]]
    x = torch.tensor(np.vstack([user_scaled, neighbor_features]), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index)

    model.eval()
    with torch.no_grad():
        pred = model(data)[0].item()

    return pred

# Load and process data
df, fuel_enc, seller_enc, trans_enc, owner_enc = load_data()
X, y, X_scaled, scaler = preprocess_features(df)

# Train-test split must be reflected in the graph as well
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
train_graph = build_knn_graph(X_train, y_train.values)
graph_hash = hash_graph(train_graph)
gnn_model, gnn_mae, gnn_r2, gnn_rmse = train_gnn_whole_graph(graph_hash, train_graph)

tabnet_model, tabnet_mae, tabnet_r2, tabnet_rmse = train_tabnet(X_train, y_train.values, X_val, y_val.values)

# Streamlit UI 
st.title("🚗 Fuel Efficiency Predictor")
st.markdown("### 📊 Model Comparison")
st.write("Compare model performance on predicting mileage")

results_df = pd.DataFrame({
    'Model': ['TabNet', 'GNN'],
    'MAE': [tabnet_mae, gnn_mae],
    'RMSE': [tabnet_rmse, gnn_rmse],
    'R² Score': [tabnet_r2, gnn_r2]
})

st.dataframe(results_df)

fig = px.bar(results_df, x='Model', y='MAE', color='Model', title='Mean Absolute Error Comparison')
st.plotly_chart(fig)

fig2 = px.bar(results_df, x='Model', y='R² Score', color='Model', title='R² Score Comparison')
st.plotly_chart(fig2)

fig_rmse = px.bar(results_df, x='Model', y='RMSE', color='Model', title='Root Mean Squared Error Comparison')
st.plotly_chart(fig_rmse)

# User Input Prediction 
st.markdown("---")
st.markdown("### 🎯 Predict Fuel Efficiency for Your Car")

st.sidebar.header("Enter Car Specifications")

year = st.sidebar.slider("Year", int(df["year"].min()), int(df["year"].max()), 2015)
km_driven = st.sidebar.slider("Kilometers Driven", 0, int(df["km_driven"].max()), 30000, step=1000)
engine = st.sidebar.slider("Engine CC", 500, 5000, 1200)
max_power = st.sidebar.slider("Max Power (bhp)", 20.0, 300.0, 70.0)

fuel_option = st.sidebar.selectbox("Fuel Type", fuel_enc.classes_)
seller_option = st.sidebar.selectbox("Seller Type", seller_enc.classes_)
trans_option = st.sidebar.selectbox("Transmission", trans_enc.classes_)
owner_option = st.sidebar.selectbox("Owner Type", owner_enc.classes_)

fuel_encoded = fuel_enc.transform([fuel_option])[0]
seller_encoded = seller_enc.transform([seller_option])[0]
trans_encoded = trans_enc.transform([trans_option])[0]
owner_encoded = owner_enc.transform([owner_option])[0]

user_input = pd.DataFrame([[
    year, km_driven, engine, max_power,
    fuel_encoded, seller_encoded, trans_encoded, owner_encoded
]], columns=['year', 'km_driven', 'engine', 'max_power', 'fuel_enc', 'seller_enc', 'trans_enc', 'owner_enc'])

user_input_scaled = scaler.transform(user_input)

# Predict using TabNet
tabnet_pred = tabnet_model.predict(user_input_scaled).flatten()[0]

# Predict using GNN
full_X_scaled = X_train  
full_y = y_train.values

gnn_pred = predict_gnn_live(gnn_model, full_X_scaled, full_y, scaler, user_input)

# Show results
st.subheader("🔍 Predicted Mileage by Both Models")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔵 TabNet")
    st.success(f"{tabnet_pred:.2f} km/l**")
    st.markdown(f"MAE: {tabnet_mae:.2f}  \nR² Score: {tabnet_r2:.2f}  \nRMSE: {tabnet_rmse:.2f}")

with col2:
    st.markdown("#### 🟢 GNN")
    st.success(f"{gnn_pred:.2f} km/l**")
    st.markdown(f"MAE: {gnn_mae:.2f}  \nR² Score: {gnn_r2:.2f}  \nRMSE: {gnn_rmse:.2f}")
