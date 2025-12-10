import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F
import pandas as pd
import os
import sys

# --- 1. SETTINGS AND PAGE LAYOUT ---
st.set_page_config(
    page_title="AI Retail Genius",
    layout="wide",
    page_icon="🛍️",
    initial_sidebar_state="expanded"
)

# Java Home Settings (Precautionary measure if the Java path is not found on the system)
# Java is generally required for Windows users.
try:
    import findspark
    findspark.init()
except:
    pass

# --- 2. SPARK SESSION (Cached) ---
@st.cache_resource
def get_spark_session():
    # Since we will be working in local mode, we make the memory settings according to your computer.
    return SparkSession.builder \
        .appName("Streamlit_Retail_App") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

# Starting a Session
try:
    spark = get_spark_session()
except Exception as e:
    st.error("Spark failed to start! Please make sure you have Java (JDK 8 or 11) installed on your computer.")
    st.error(f"Error Detail: {e}")
    st.stop()

# --- 3. DATA LOADING ---
@st.cache_resource
def load_assets():
    model_path = "als_quantity_model"
    products_path = "product_names.parquet"
    segments_path = "customer_segments.parquet"
    mapping_path = "item_mapping.parquet"
    
    if not os.path.exists(model_path):
        return None, None, None, None, "Model folder not found."
        
    try:
        model = ALSModel.load(model_path)
        products = spark.read.parquet(products_path).cache()
        segments = spark.read.parquet(segments_path).cache()
        mapping = spark.read.parquet(mapping_path).cache()
        return model, products, segments, mapping, None
    except Exception as e:
        return None, None, None, None, str(e)


with st.spinner('Loading Artificial Intelligence Model and Data...'):
    model, df_products, df_segments, df_mapping, error_msg = load_assets()

if error_msg:
    st.error(f"File Upload Error: {error_msg}")
    st.warning("Please make sure that 'als_quantity_model' and parquet folders are next to each other in app.py.")
    st.stop()

# --- 4. INTERFACE DESIGN ---

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
    st.title("Retail AI")
    st.markdown("This application uses the **PySpark ALS** algorithm to make personalized product and quantity predictions.")
    st.markdown("---")
    st.info("**Hint:** Since the model is trained with logarithmic transformation, the results are presented in terms of 'Actual Quantity' by performing the inverse transformation (Exp-1).")


st.title("Customer Analysis and Sales Forecast System")
st.markdown("Enter your Customer ID, the system will analyze your **RFM Segment** and estimate how many units of each product you can sell.")

col_input, col_space = st.columns([1, 2])
with col_input:
    user_input = st.number_input("Enter Customer ID:", min_value=1, value=12347, step=1, help="Customer ID in the dataset")
    analyze_btn = st.button("Analyze it ! ", type="primary", use_container_width=True)

if analyze_btn:
    
    cust_data_row = df_segments.filter(F.col("Customer ID") == user_input).first()
    
    if cust_data_row:
        st.divider()
        st.subheader(f"Customer Profile: {user_input}")
        
        cluster_id = cust_data_row["Cluster"]
        cluster_map = {
            0: "Standard/Potantially",
            1: "Whales",
            2: "VIP",
            3: "Hibernating/Churned",
            4: "Loyal Champions"
        }
        segment_label = cluster_map.get(cluster_id, f"Segment {cluster_id}")
        
      
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Transaction (Recency)", f"{cust_data_row['Recency']} Days")
        c2.metric("Frequency (Frequency)", f"{cust_data_row['Frequency']} Order")
        c3.metric("Total Income (Monetary)", f"{cust_data_row['Monetary']:.0f} ")
        
        
        if cluster_id in [1, 2]:
            c4.error(f"✨ {segment_label}") 
        elif cluster_id == 3:
            c4.success(f"⭐ {segment_label}") 
        else:
            c4.info(f"🔹 {segment_label}") 
        
        
        st.subheader("Sales Prediction : ")
        
        user_df = spark.createDataFrame([(int(user_input),)], ["user_id"])
        
        raw_recs = model.recommendForUserSubset(user_df, 10)
        
        if raw_recs.count() > 0:
            recs_exploded = raw_recs.select(F.explode("recommendations").alias("rec"))
            
            final_predictions = recs_exploded.select(
                F.col("rec.item_id"),
                (F.exp(F.col("rec.rating")) - 1).alias("predicted_qty")
            ) \
            .join(df_mapping, on="item_id") \
            .join(df_products, on="StockCode") \
            .select("StockCode", "Description", "predicted_qty") \
            .orderBy(F.col("predicted_qty").desc()) \
            .limit(10)
            
            pandas_df = final_predictions.toPandas()
            
            pandas_df["predicted_qty"] = pandas_df["predicted_qty"].apply(lambda x: f"{x:.2f} Adet")
            pandas_df.columns = ["Stock Code", "Product Name", "Predicted Order Qty"]
            
            st.dataframe(pandas_df, use_container_width=True, hide_index=True)
            
            top_prediction_val = float(pandas_df.iloc[0]["Predicted Order Qty"].split()[0])
            if top_prediction_val > 2.0:
                st.success(f"**Opportunity:** This customer `{pandas_df.iloc[0]['Product Name']}` You can make multiple purchases from the product! Package offers may be offered.")
            elif top_prediction_val < 0.5:
                st.info("**Note:** Estimated quantities are low. These products can be suggested as 'basket completion' items.")
                
        else:
            st.warning("There is not enough data for this user (Cold Start).")
            
    else:
        st.error(f"❌ ID {user_input} was not found in the database. Please try another ID.")


st.markdown("---")
st.caption("Developed with PySpark & Streamlit")