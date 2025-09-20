import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "/content/Last mile Delivery Data.csv"  # Update path
df = pd.read_csv(file_path)

# Drop rows with critical missing data
df.dropna(subset=['Order_ID', 'Delivery_Time', 'Vehicle'], inplace=True)

# Fill categorical NAs
categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Fill numeric NAs with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Age groups for scatter plot
df['AgentAgeGroup'] = pd.cut(
    df['Agent_Age'],
    bins=[0, 25, 40, 100],
    labels=['<25', '25–40', '40+']
)

sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# -------- 1. Delay Analyzer --------
plt.figure(figsize=(10,6))
sns.barplot(
    data=df,
    x="Weather",
    y="Delivery_Time",
    hue="Traffic",
    ci="sd"
)
plt.title("Delay Analyzer: Avg Delivery Time by Weather & Traffic")
plt.ylabel("Average Delivery Time (mins)")
plt.xlabel("Weather Condition")
plt.legend(title="Traffic")
plt.tight_layout()
plt.savefig("delay_analyzer.png")
plt.close()

# -------- 2. Vehicle Comparison --------
plt.figure(figsize=(8,6))
sns.barplot(
    data=df,
    x="Vehicle",
    y="Delivery_Time",
    ci="sd"
)
plt.title("Vehicle Comparison: Avg Delivery Time by Vehicle")
plt.ylabel("Average Delivery Time (mins)")
plt.xlabel("Vehicle Type")
plt.tight_layout()
plt.savefig("vehicle_comparison.png")
plt.close()

# -------- 3. Agent Performance Scatter --------
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="Agent_Rating",
    y="Delivery_Time",
    hue="AgentAgeGroup",
    palette="deep"
)
plt.title("Agent Performance: Rating vs Delivery Time (by Age Group)")
plt.ylabel("Delivery Time (mins)")
plt.xlabel("Agent Rating")
plt.legend(title="Age Group")
plt.tight_layout()
plt.savefig("agent_performance_scatter.png")
plt.close()

# -------- 4. Area Heatmap --------
plt.figure(figsize=(10,6))
area_pivot = df.groupby("Area")["Delivery_Time"].mean().reset_index()
area_pivot = area_pivot.pivot_table(values="Delivery_Time", index="Area", aggfunc=np.mean)

sns.heatmap(
    area_pivot,
    annot=True,
    cmap="YlOrRd",
    cbar_kws={'label': 'Avg Delivery Time (mins)'}
)
plt.title("Area Heatmap: Avg Delivery Time by Area")
plt.ylabel("Area")
plt.xlabel("")
plt.tight_layout()
plt.savefig("area_heatmap.png")
plt.close()

# -------- 5. Category Visualizer --------
plt.figure(figsize=(10,6))
sns.boxplot(
    data=df,
    x="Category",
    y="Delivery_Time"
)
plt.title("Category Visualizer: Delivery Time Distribution by Product Category")
plt.ylabel("Delivery Time (mins)")
plt.xlabel("Product Category")
plt.tight_layout()
plt.savefig("category_boxplot.png")
plt.close()

# (a) Monthly Trend Line Chart
if "DeliveryDate" in df.columns:
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    df["Month"] = df["DeliveryDate"].dt.to_period("M")
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df.groupby("Month")["Delivery_Time"].mean().reset_index(),
        x="Month",
        y="Delivery_Time",
        marker="o"
    )
    plt.title("Monthly Trend: Avg Delivery Time")
    plt.ylabel("Average Delivery Time (mins)")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("monthly_trend.png")
    plt.close()

# (b) Delivery Time Distribution
plt.figure(figsize=(8,6))
sns.histplot(df["Delivery_Time"], kde=True, bins=30)
plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (mins)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("delivery_time_distribution.png")
plt.close()

# (c) % of Late Deliveries by Traffic
avg_time = df["Delivery_Time"].mean()
threshold = avg_time + df["Delivery_Time"].std()
df["LateDeliveryFlag"] = np.where(df["Delivery_Time"] > threshold, 1, 0)

late_summary = df.groupby("Traffic")["LateDeliveryFlag"].mean().reset_index()
late_summary["LateDeliveryFlag"] *= 100

plt.figure(figsize=(8,6))
sns.barplot(data=late_summary, x="Traffic", y="LateDeliveryFlag")
plt.title("% of Late Deliveries by Traffic")
plt.ylabel("Late Deliveries (%)")
plt.xlabel("Traffic Condition")
plt.tight_layout()
plt.savefig("late_deliveries_by_traffic.png")
plt.close()

# (d) Agent Count per Area
plt.figure(figsize=(10,6))
sns.countplot(data=df, x="Area", order=df["Area"].value_counts().index)
plt.title("Agent Count per Area")
plt.ylabel("Number of Agents")
plt.xlabel("Area")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("agent_count_per_area.png")
plt.close()

print("\n✅ All compulsory + optional visuals generated and saved as PNGs!")
