import os
import shutil
import threading
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType, ArrayType
from pyspark.sql.functions import explode, col, sum, count, window, approx_count_distinct, struct, collect_set, from_json, avg, round, to_timestamp, to_json

shutil.rmtree('checkpoints', ignore_errors=True)
if os.path.exists('revenue.csv'): os.remove('revenue.csv')
if os.path.exists('fraud.csv'): os.remove('fraud.csv')

spark_version = pyspark.__version__
scala_ver = "2.13" if spark_version.startswith("4") else "2.12"
spark = SparkSession.builder \
    .appName("Ashpaz_RealTime_Pipeline") \
    .master("local[*]") \
    .config("spark.jars.packages", f"org.apache.spark:spark-sql-kafka-0-10_{scala_ver}:{spark_version}") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

item_schema = StructType([
    StructField("category", StringType(), True),
    StructField("food_item", StringType(), True),
    StructField("quantity", LongType(), True),
    StructField("unit_price", DoubleType(), True)
])

order_schema = StructType([
    StructField("cuisines", ArrayType(StringType()), True),
    StructField("items", ArrayType(item_schema), True),
    StructField("order_id", StringType(), True),
    StructField("order_price", DoubleType(), True),
    StructField("order_time", StringType(), True),
    StructField("phone_number", StringType(), True),
    StructField("request_online", BooleanType(), True),
    StructField("request_table", BooleanType(), True),
    StructField("restaurant_city", StringType(), True),
    StructField("restaurant_id", LongType(), True),
    StructField("restaurant_name", StringType(), True),
    StructField("user_id", StringType(), True)
])

parsed_stream_df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "ashpaz.valid") \
    .option("startingOffsets", "latest") \
    .load() \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), order_schema).alias("data")) \
    .select("data.*") \
    .withColumn("timestamp", to_timestamp(col("order_time")))

geo_fraud_df = parsed_stream_df.withWatermark("timestamp", "10 minutes") \
    .groupBy(col("user_id"), window(col("timestamp"), "30 minutes")) \
    .agg(approx_count_distinct("restaurant_city").alias("unique_cities"), collect_set("restaurant_city").alias("city_list")) \
    .filter(col("unique_cities") > 1) \
    .selectExpr("user_id", "window.start AS window_start", "window.end AS window_end", "'Geographical_Impossibility' AS fraud_type", "CAST(city_list AS STRING) AS details")

vel_fraud_df = parsed_stream_df.withWatermark("timestamp", "10 minutes") \
    .groupBy(col("user_id"), window(col("timestamp"), "60 minutes")) \
    .agg(count("order_id").alias("order_count")) \
    .filter(col("order_count") > 5) \
    .selectExpr("user_id", "window.start AS window_start", "window.end AS window_end", "'Velocity_Spam' AS fraud_type", "CAST(order_count AS STRING) AS details")

all_alerts_df = geo_fraud_df.union(vel_fraud_df)

live_cuisine_df = parsed_stream_df.withColumn("cuisine", explode(col("cuisines"))) \
    .groupBy("cuisine") \
    .agg(round(sum("order_price"), 2).alias("total_revenue"), round(avg("order_price"), 2).alias("avg_order_value"))

def process_fraud_batch(df, epoch_id):
    df.show(truncate=False)
    pdf = df.groupBy("fraud_type").count().toPandas()
    if not pdf.empty:
        pdf.to_csv("fraud.csv", index=False)

def process_revenue_batch(df, epoch_id):
    df.show()
    pdf = df.orderBy(col("total_revenue").desc()).limit(10).toPandas()
    if not pdf.empty:
        pdf.to_csv("revenue.csv", index=False)

all_alerts_df.selectExpr("user_id AS key", "to_json(struct(user_id, window_start, window_end, fraud_type, details)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "ashpaz.error_log") \
    .option("checkpointLocation", "checkpoints/fraud_alerts") \
    .outputMode("append") \
    .trigger(processingTime='1 seconds') \
    .start()

all_alerts_df.writeStream \
    .foreachBatch(process_fraud_batch) \
    .outputMode("update") \
    .trigger(processingTime='1 seconds') \
    .start()

live_cuisine_df.writeStream \
    .foreachBatch(process_revenue_batch) \
    .outputMode("complete") \
    .trigger(processingTime='1 seconds') \
    .start()

app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#f4f6f9', 'minHeight': '100vh', 'padding': '30px', 'fontFamily': '"Segoe UI", Roboto, Helvetica, Arial, sans-serif'}, children=[
    html.Div(style={'backgroundColor': '#ffffff', 'padding': '20px 40px', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)', 'marginBottom': '30px', 'textAlign': 'center'}, children=[
        html.H1("Ashpaz Live Intelligence Dashboard", style={'color': '#1a365d', 'margin': '0', 'fontWeight': '600', 'letterSpacing': '1px'}),
        html.P("Real-Time Streaming Analytics & Security Operations", style={'color': '#718096', 'margin': '10px 0 0 0', 'fontSize': '16px'})
    ]),
    html.Div(style={'display': 'flex', 'gap': '30px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={'flex': '1', 'minWidth': '45%', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'padding': '25px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}, children=[
            html.H3("Active Security Alerts", style={'color': '#2d3748', 'marginTop': '0', 'borderBottom': '2px solid #edf2f7', 'paddingBottom': '10px'}),
            dcc.Graph(id='fraud-graph', config={'displayModeBar': False})
        ]),
        html.Div(style={'flex': '1', 'minWidth': '45%', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'padding': '25px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'}, children=[
            html.H3("Live Revenue Analytics", style={'color': '#2d3748', 'marginTop': '0', 'borderBottom': '2px solid #edf2f7', 'paddingBottom': '10px'}),
            dcc.Graph(id='revenue-graph', config={'displayModeBar': False})
        ])
    ]),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@app.callback(
    [Output('fraud-graph', 'figure'), Output('revenue-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    if os.path.exists("fraud.csv") and os.path.getsize("fraud.csv") > 0:
        try:
            fraud_pdf = pd.read_csv("fraud.csv")
            fig_fraud = px.bar(fraud_pdf, x='fraud_type', y='count', color='fraud_type',
                               labels={'fraud_type': 'Alert Type', 'count': 'Total Flags'},
                               color_discrete_map={'Geographical_Impossibility': '#e53e3e', 'Velocity_Spam': '#dd6b20'},
                               template='plotly_white')
            fig_fraud.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig_fraud.update_traces(marker_line_width=0)
        except Exception:
            fig_fraud = px.bar(title="Reading fraud data...", template='plotly_white')
    else:
        fig_fraud = px.bar(title="Monitoring... No fraud detected yet.", template='plotly_white')
        fig_fraud.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    if os.path.exists("revenue.csv") and os.path.getsize("revenue.csv") > 0:
        try:
            revenue_pdf = pd.read_csv("revenue.csv")
            fig_rev = make_subplots(specs=[[{"secondary_y": True}]])
            fig_rev.add_bar(x=revenue_pdf['cuisine'], y=revenue_pdf['total_revenue'], name="Total Revenue", marker_color='#3182ce')
            fig_rev.add_trace(px.line(revenue_pdf, x='cuisine', y='avg_order_value', markers=True).data[0], secondary_y=True)
            fig_rev.data[1].line.color = '#38a169'
            fig_rev.data[1].line.width = 3
            fig_rev.data[1].name = "Avg Order Value"
            fig_rev.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=30, b=20), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig_rev.update_yaxes(title_text="Total Revenue ($)", secondary_y=False, gridcolor='#edf2f7')
            fig_rev.update_yaxes(title_text="Avg Order Value ($)", secondary_y=True, showgrid=False)
            fig_rev.update_traces(marker_line_width=0, selector=dict(type="bar"))
        except Exception:
            fig_rev = px.bar(title="Reading revenue data...", template='plotly_white')
    else:
        fig_rev = px.bar(title="Monitoring... Waiting for revenue data.", template='plotly_white')
        fig_rev.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig_fraud, fig_rev

def run_dash():
    app.run(debug=False, use_reloader=False, port=8050)

dash_thread = threading.Thread(target=run_dash)
dash_thread.daemon = True
dash_thread.start()

spark.streams.awaitAnyTermination()