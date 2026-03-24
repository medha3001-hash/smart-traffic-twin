import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load the data ----
df = pd.read_csv('data/traffic.csv')

# ---- Fix the DateTime column ----
# Right now DateTime is plain text. We convert it to a real date object
# so Python can understand "this is 8am" or "this is a Monday"
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract useful time features from the DateTime column
df['hour'] = df['DateTime'].dt.hour        # 0 to 23
df['day_of_week'] = df['DateTime'].dt.dayofweek  # 0=Monday, 6=Sunday
df['month'] = df['DateTime'].dt.month      # 1 to 12

print("Data loaded and time features extracted!")
print(df.head())

# ---- Set a nice visual style for all graphs ----
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

# ============================================================
# GRAPH 1: Average vehicles per hour (traffic pattern by hour)
# ============================================================
hourly_avg = df.groupby(['hour', 'Junction'])['Vehicles'].mean().reset_index()

plt.figure()
for junction in [1, 2, 3, 4]:
    data = hourly_avg[hourly_avg['Junction'] == junction]
    plt.plot(data['hour'], data['Vehicles'], marker='o', label=f'Junction {junction}')

plt.title('Average Vehicle Count by Hour of Day')
plt.xlabel('Hour (0 = midnight, 12 = noon)')
plt.ylabel('Average Vehicles')
plt.legend()
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('data/graph1_hourly_traffic.png')
plt.show()
print("Graph 1 saved!")

# ============================================================
# GRAPH 2: Average vehicles by day of week
# ============================================================
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_avg = df.groupby('day_of_week')['Vehicles'].mean().reset_index()
daily_avg['day_name'] = daily_avg['day_of_week'].apply(lambda x: day_names[x])

plt.figure()
sns.barplot(data=daily_avg, x='day_name', y='Vehicles', palette='Blues_d')
plt.title('Average Vehicle Count by Day of Week')
plt.xlabel('Day')
plt.ylabel('Average Vehicles')
plt.tight_layout()
plt.savefig('data/graph2_daily_traffic.png')
plt.show()
print("Graph 2 saved!")

# ============================================================
# GRAPH 3: Vehicle count distribution per junction
# ============================================================
plt.figure()
for junction in [1, 2, 3, 4]:
    data = df[df['Junction'] == junction]['Vehicles']
    sns.kdeplot(data, label=f'Junction {junction}', fill=True, alpha=0.3)

plt.title('Distribution of Vehicle Counts per Junction')
plt.xlabel('Number of Vehicles')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('data/graph3_distribution.png')
plt.show()
print("Graph 3 saved!")

print("\n✅ All 3 graphs created and saved in the data/ folder!")