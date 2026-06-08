# 📊 Zomato Data Analysis — Complete Technical Concepts Guide

An in-depth reference covering every **Python**, **Data Science**, **Statistics**, and **Machine Learning** concept relevant to the Zomato Dataset Analysis project — including concepts like One-Hot Encoding that extend naturally from this project.

---

## 📚 Table of Contents

1. [Project Goal](#1-project-goal)
2. [Libraries Used](#2-libraries-used)
3. [Dataset Overview & Data Types](#3-dataset-overview--data-types)
4. [Core Data Science Concepts](#4-core-data-science-concepts)
   - [Exploratory Data Analysis (EDA)](#41-exploratory-data-analysis-eda)
   - [Feature Engineering](#42-feature-engineering)
   - [Data Types: Categorical vs Numerical](#43-data-types-categorical-vs-numerical)
   - [One-Hot Encoding](#44-one-hot-encoding)
   - [Label Encoding](#45-label-encoding)
   - [Missing Data & Imputation](#46-missing-data--imputation)
   - [Outliers](#47-outliers)
   - [Data Normalization & Standardization](#48-data-normalization--standardization)
   - [Correlation](#49-correlation)
5. [Statistical Concepts](#5-statistical-concepts)
   - [Mean, Median, Mode](#51-mean-median-mode)
   - [Variance & Standard Deviation](#52-variance--standard-deviation)
   - [Percentiles & Quartiles (IQR)](#53-percentiles--quartiles-iqr)
   - [Distribution & Skewness](#54-distribution--skewness)
   - [Frequency & Frequency Distribution](#55-frequency--frequency-distribution)
6. [Data Pre-Processing in This Project](#6-data-pre-processing-in-this-project)
   - [Reading a CSV File](#61-reading-a-csv-file)
   - [Inspecting the DataFrame](#62-inspecting-the-dataframe)
   - [Dropping Columns (Feature Selection)](#63-dropping-columns-feature-selection)
   - [Dropping Duplicates](#64-dropping-duplicates)
   - [Cleaning the `rate` Column (Data Parsing)](#65-cleaning-the-rate-column-data-parsing)
   - [Handling Null Values (Imputation)](#66-handling-null-values-imputation)
   - [Renaming Columns](#67-renaming-columns)
   - [Cleaning the `Cost2plates` Column](#68-cleaning-the-cost2plates-column)
   - [Clustering Rare Categories (Binning)](#69-clustering-rare-categories-binning)
7. [Data Visualization Concepts](#7-data-visualization-concepts)
   - [Count Plot](#71-count-plot)
   - [Box Plot](#72-box-plot)
   - [Bar Plot / Grouped Bar Plot](#73-bar-plot--grouped-bar-plot)
   - [Heatmap](#74-heatmap)
   - [Pivot Table](#75-pivot-table)
8. [Key Python Concepts Used](#8-key-python-concepts-used)
   - [Custom Functions (def)](#81-custom-functions-def)
   - [apply()](#82-apply)
   - [value_counts()](#83-value_counts)
   - [Boolean Indexing / Filtering](#84-boolean-indexing--filtering)
   - [groupby()](#85-groupby)
9. [ML Concepts: What Comes Next?](#9-ml-concepts-what-comes-next)
   - [One-Hot Encoding (Deep Dive)](#91-one-hot-encoding-deep-dive)
   - [Train/Test Split](#92-traintest-split)
   - [Supervised vs Unsupervised Learning](#93-supervised-vs-unsupervised-learning)
10. [Key Business Insights](#10-key-business-insights)
11. [Full Workflow Summary](#11-full-workflow-summary)

---

## 1. Project Goal

A client wants to open a new restaurant and needs **data-driven** answers to three questions:

| # | Business Question |
|---|-------------------|
| 1 | What **type** of restaurant should be opened? |
| 2 | What is the **best location**? |
| 3 | What **services** (online ordering, table booking) should it offer? |

Dataset source: **Kaggle** — real Zomato restaurant listings from **Bangalore, India**.

---

## 2. Libraries Used

### 🐼 Pandas (`pandas`)
> The core library for structured data manipulation. Introduces the **DataFrame** concept — a 2D table of rows and columns.

```python
import pandas as pd
```

| Function | Purpose |
|----------|---------|
| `pd.read_csv()` | Load CSV into DataFrame |
| `df.head()` | Preview first 5 rows |
| `df.info()` | Column types, null counts |
| `df.describe()` | Summary statistics |
| `df.drop()` | Remove columns/rows |
| `df.rename()` | Rename columns |
| `df.dropna()` | Remove rows with nulls |
| `df.fillna()` | Fill nulls with a value |
| `df.groupby()` | Group rows for aggregation |
| `df.apply()` | Apply a function element-wise |
| `df.pivot_table()` | Reshape into summary table |

---

### 🔢 NumPy (`numpy`)
> Foundation for numerical computing. Provides n-dimensional arrays and math functions.

```python
import numpy as np
```

Used here primarily for `np.nan` — the standard representation of a **missing value**.

---

### 📈 Matplotlib (`matplotlib`)
> The base plotting library. Provides the canvas, figure, and axes objects that all other plot libraries build upon.

```python
import matplotlib.pyplot as plt
plt.style.use('dark_background')
```

---

### 🎨 Seaborn (`seaborn`)
> High-level statistical plotting library built on top of Matplotlib. Produces beautiful statistical charts with minimal code.

```python
import seaborn as sns
```

---

## 3. Dataset Overview & Data Types

The raw dataset has **51,717 rows** and **17 columns**.

| Column | Data Type | Kind |
|--------|-----------|------|
| `url` | String | Categorical (nominal) |
| `name` | String | Categorical (nominal) |
| `online_order` | String (`Yes`/`No`) | Categorical (binary) |
| `book_table` | String (`Yes`/`No`) | Categorical (binary) |
| `rate` | String → Float (after cleaning) | Numerical (continuous) |
| `votes` | Integer | Numerical (discrete) |
| `location` | String | Categorical (nominal) |
| `rest_type` | String | Categorical (nominal) |
| `cuisines` | String | Categorical (nominal) |
| `approx_cost(for two people)` | String → Float (after cleaning) | Numerical (continuous) |
| `listed_in(type)` | String | Categorical (nominal) |

---

## 4. Core Data Science Concepts

### 4.1 Exploratory Data Analysis (EDA)

> **EDA** is the process of examining a dataset to understand its structure, patterns, relationships, and anomalies **before** applying any model or making decisions.

EDA answers questions like:
- What does the data look like?
- Are there missing values?
- What are the distributions of each variable?
- Are there relationships between variables?

**This entire Zomato project is an EDA exercise.** No machine learning model is built — the goal is purely to extract insights through cleaning and visualisation.

**EDA Checklist:**
```
✅ View shape and column names
✅ Check data types
✅ Count missing values
✅ Look at value distributions (value_counts, describe)
✅ Visualize distributions (histograms, boxplots)
✅ Visualize relationships (bar plots, heatmaps)
✅ Identify and handle outliers
✅ Identify and handle missing data
```

---

### 4.2 Feature Engineering

> **Feature Engineering** is the process of transforming raw data into meaningful inputs (features) for analysis or modeling.

In this project:

| Raw Feature | Engineered Feature | Transformation |
|-------------|-------------------|----------------|
| `"4.1/5"` | `4.1` (float) | Parsing + type casting |
| `"1,200"` | `1200.0` (float) | String cleaning |
| `"Casual Dining, Cafe"` | `"others"` | Frequency-based binning |
| `approx_cost(for two people)` | `Cost2plates` | Renaming |

---

### 4.3 Data Types: Categorical vs Numerical

Understanding data types is **fundamental** to choosing the right analysis technique.

#### Categorical Data
> Data that represents groups or categories. Cannot do arithmetic on it.

| Sub-type | Description | Example |
|----------|-------------|---------|
| **Nominal** | No order between categories | `location`, `rest_type`, `cuisines` |
| **Ordinal** | Has a meaningful order | Rating levels (Low, Medium, High) |
| **Binary** | Only two values | `online_order` (Yes/No) |

#### Numerical Data
> Data that represents measurable quantities. You can do arithmetic on it.

| Sub-type | Description | Example |
|----------|-------------|---------|
| **Continuous** | Any value in a range | `rate`, `Cost2plates` |
| **Discrete** | Only whole numbers | `votes` |

> ⚠️ **Why this matters:** You cannot compute the "mean location" — that's categorical. But you CAN compute the mean rating — that's numerical. Mixing these up leads to wrong analysis.

---

### 4.4 One-Hot Encoding

> **One-Hot Encoding (OHE)** converts a categorical column into **multiple binary (0/1) columns** — one per unique category.

#### Why is it needed?
Machine learning algorithms work with **numbers only**. If you have a column like `online_order` with values `"Yes"` and `"No"`, you must convert it to numbers. But simply encoding `Yes=1, No=0` introduces a false numeric relationship. One-Hot Encoding avoids this.

#### Example

Original data:
```
restaurant | online_order
-----------|-------------
A          | Yes
B          | No
C          | Yes
```

After One-Hot Encoding:
```
restaurant | online_order_Yes | online_order_No
-----------|-----------------|----------------
A          | 1               | 0
B          | 0               | 1
C          | 1               | 0
```

Each category becomes its **own binary column**. Each row has exactly one `1` and the rest `0`s.

#### In Python with Pandas:
```python
df_encoded = pd.get_dummies(df, columns=['online_order', 'book_table'])
```

#### In Python with Scikit-learn:
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['rest_type']])
```

#### Dummy Variable Trap
When using OHE, you should drop **one column** per category to avoid perfect multicollinearity (where one column is completely predictable from the others):

```python
pd.get_dummies(df, columns=['online_order'], drop_first=True)
# Now only 'online_order_Yes' exists — 'No' is implied when it's 0
```

#### When to apply OHE in this project:
OHE is **not used** in this EDA project. However, if you wanted to build a **predictive model** (e.g., predict a restaurant's rating from its features), you would apply OHE to columns like:
- `online_order` → `online_order_Yes`
- `book_table` → `book_table_Yes`
- `rest_type` → `rest_type_Quick_Bites`, `rest_type_Casual_Dining`, ...
- `location` → one column per location

---

### 4.5 Label Encoding

> **Label Encoding** assigns a unique integer to each category.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['online_order_encoded'] = le.fit_transform(df['online_order'])
# 'No' → 0, 'Yes' → 1
```

**⚠️ Problem:** Label Encoding implies an order (`No < Yes`). This is only appropriate for **ordinal** categories. For nominal categories, use **One-Hot Encoding** instead.

| Method | Use When |
|--------|----------|
| One-Hot Encoding | Nominal categories (no order) |
| Label Encoding | Ordinal categories (e.g., Low=0, Medium=1, High=2) |
| Binary Encoding | High-cardinality categories (many unique values) |

---

### 4.6 Missing Data & Imputation

> **Missing data** (null / NaN values) occurs when a value is not recorded for a particular row.

#### Types of Missingness:
| Type | Description |
|------|-------------|
| **MCAR** (Missing Completely At Random) | The missingness has no pattern |
| **MAR** (Missing At Random) | Missingness depends on other observed data |
| **MNAR** (Missing Not At Random) | Missingness depends on the missing value itself |

#### Strategies used in this project:

| Column | Strategy | Reason |
|--------|----------|--------|
| `rate` | **Mean imputation** (`fillna(mean)`) | ~20% missing; mean is a safe neutral value |
| `location`, `cuisines`, `rest_type` | **Row deletion** (`dropna()`) | Very few nulls (<1%); safe to remove |

#### Other Common Strategies (not used here):
```python
df['col'].fillna(df['col'].median())   # Median imputation (better for skewed data)
df['col'].fillna(df['col'].mode()[0])  # Mode imputation (best for categorical)
df['col'].fillna(method='ffill')       # Forward fill (use previous row's value)
df['col'].fillna(method='bfill')       # Backward fill (use next row's value)
```

---

### 4.7 Outliers

> **Outliers** are data points that differ significantly from other observations.

#### Detection using IQR (Interquartile Range):
```python
Q1 = df['rate'].quantile(0.25)
Q3 = df['rate'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['rate'] < lower_bound) | (df['rate'] > upper_bound)]
```

#### Visualizing outliers — Box Plot:
Box plots (used in this project) directly show outliers as dots beyond the whiskers.

#### Handling Outliers:
| Approach | When |
|----------|------|
| **Keep** | If they represent real data (genuine extreme restaurants) |
| **Remove** | If they're data entry errors |
| **Cap (Winsorize)** | Replace extremes with boundary values |

---

### 4.8 Data Normalization & Standardization

> These techniques rescale numerical features so they're comparable.

#### Normalization (Min-Max Scaling)
Scales values to the range **[0, 1]**:

```
X_normalized = (X - X_min) / (X_max - X_min)
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['rate_normalized'] = scaler.fit_transform(df[['rate']])
```

#### Standardization (Z-Score Scaling)
Scales values so the mean = 0 and standard deviation = 1:

```
Z = (X - mean) / std_dev
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['rate_standardized'] = scaler.fit_transform(df[['rate']])
```

| Method | Use When |
|--------|----------|
| Normalization | Data is NOT normally distributed; used in neural nets, KNN |
| Standardization | Data is approximately normal; used in linear/logistic regression, SVM |

> **Not applied in this project** (EDA doesn't require it), but essential before training ML models.

---

### 4.9 Correlation

> **Correlation** measures how strongly two variables move together.

- **Positive correlation (+1)**: As X goes up, Y goes up
- **Negative correlation (-1)**: As X goes up, Y goes down
- **No correlation (0)**: No relationship

```python
df[['rate', 'votes', 'Cost2plates']].corr()
```

Visualize as a heatmap:
```python
sns.heatmap(df[['rate', 'votes', 'Cost2plates']].corr(), annot=True, cmap='coolwarm')
```

> ⚠️ **Correlation ≠ Causation.** A high correlation between two variables doesn't mean one causes the other.

---

## 5. Statistical Concepts

### 5.1 Mean, Median, Mode

| Measure | Description | Formula | Python |
|---------|-------------|---------|--------|
| **Mean** | Average of all values | `Σx / n` | `df['rate'].mean()` |
| **Median** | Middle value when sorted | — | `df['rate'].median()` |
| **Mode** | Most frequent value | — | `df['rate'].mode()[0]` |

**Which to use for missing value imputation?**
- **Mean**: Good for symmetric (normal) distributions
- **Median**: Better when data is **skewed** or has outliers (not affected by extremes)
- **Mode**: Best for **categorical** data

**Example from this project:**
```python
df['rate'].fillna(df['rate'].mean(), inplace=True)
# Mean rating ≈ 3.7 fills the ~10,000 missing ratings
```

---

### 5.2 Variance & Standard Deviation

> **Variance** measures how spread out the data is from the mean.
> **Standard Deviation** (σ) is the square root of variance — easier to interpret (same unit as data).

```
Variance  = Σ(x - mean)² / n
Std Dev   = √Variance
```

```python
df['rate'].var()  # Variance
df['rate'].std()  # Standard Deviation
```

- **Low std dev** → data points cluster close to the mean
- **High std dev** → data points are spread out widely

---

### 5.3 Percentiles & Quartiles (IQR)

> A **percentile** tells you what percentage of the data falls below a given value.

| Quartile | Percentile | Meaning |
|----------|-----------|---------|
| Q1 | 25th | 25% of data is below this |
| Q2 | 50th | This is the Median |
| Q3 | 75th | 75% of data is below this |

**IQR (Interquartile Range):**
```
IQR = Q3 - Q1
```
The middle 50% of the data lives in this range. Box plots display this.

```python
df['rate'].quantile(0.25)  # Q1
df['rate'].quantile(0.50)  # Median (Q2)
df['rate'].quantile(0.75)  # Q3
```

---

### 5.4 Distribution & Skewness

> A **distribution** describes how data values are spread across the range.

#### Normal (Bell Curve) Distribution:
```
       |   *
       |  ***
       | *****
       |*******
       ----------
     mean=median=mode
```
Symmetric. Mean ≈ Median ≈ Mode.

#### Right-Skewed (Positive Skew):
```
       |*
       |**
       |****
       |**********
       ---------------
     Mean > Median
```
Tail extends to the **right**. Most values are low, a few are very high (e.g., `votes` — most restaurants have few votes, a few have thousands).

#### Left-Skewed (Negative Skew):
```
          *|
         **|
       ****|
  **********|
  ---------------
     Mean < Median
```
Tail extends to the **left**.

```python
df['votes'].skew()  # Positive value = right skewed
```

> **Why it matters:** Skewed distributions influence which imputation strategy and which statistical tests to use.

---

### 5.5 Frequency & Frequency Distribution

> **Frequency** = how often a value appears in the dataset.

```python
df['rest_type'].value_counts()
# Quick Bites      19010
# Casual Dining    10253
# Cafe              3682
# ...
```

> **Frequency Distribution** = the full table of all values and their frequencies.

This is used extensively in this project to:
1. Understand which restaurant types dominate
2. Decide which rare categories to merge into "others"

---

## 6. Data Pre-Processing in This Project

### 6.1 Reading a CSV File

```python
df = pd.read_csv('zomato.csv')
df.head()
```

> `pd.read_csv()` parses a comma-separated values file into a DataFrame. `df.head()` shows the first 5 rows for a quick sanity check.

---

### 6.2 Inspecting the DataFrame

```python
df.shape       # (51717, 17)
df.columns     # All column names
df.info()      # Dtypes + null counts
df.describe()  # Mean, std, min, max, quartiles for numeric columns
df['rate'].unique()  # All unique values
```

---

### 6.3 Dropping Columns (Feature Selection)

```python
df = df.drop(['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list'], axis=1)
```

> **Feature Selection** is the process of keeping only the most relevant variables. Irrelevant or redundant columns add noise and memory overhead.

**Dropped columns and why:**

| Column | Reason for dropping |
|--------|-------------------|
| `url` | Not analytically useful |
| `address` | Redundant with `location` |
| `phone` | Not analytically useful |
| `menu_item` | Mostly empty |
| `dish_liked` | Not needed for this business question |
| `reviews_list` | Free text — needs NLP; out of scope |

---

### 6.4 Dropping Duplicates

```python
df.drop_duplicates(inplace=True)
# 51,717 → 51,609 rows
```

> Duplicate rows occur when a restaurant appears more than once with identical data. They inflate counts and bias statistics.

---

### 6.5 Cleaning the `rate` Column (Data Parsing)

**Problem:** Raw values: `"4.1/5"`, `"NEW"`, `"-"`, `NaN`

```python
def handlerate(value):
    if value == 'NEW' or value == '-':
        return np.nan
    else:
        value = str(value).split('/')
        return float(value[0])

df['rate'] = df['rate'].apply(handlerate)
```

> **Data Parsing** = extracting structured information from unstructured or semi-structured strings.
>
> - `str.split('/')` → splits `"4.1/5"` into `["4.1", "5"]`
> - `float()` → converts `"4.1"` to the number `4.1`
> - `np.nan` → marks `"NEW"` and `"-"` as missing

---

### 6.6 Handling Null Values (Imputation)

```python
# Fill missing ratings with the mean
df['rate'].fillna(df['rate'].mean(), inplace=True)

# Drop rows where location/cuisines/rest_type are null
df.dropna(inplace=True)
```

See [Section 4.6](#46-missing-data--imputation) for the full theory.

---

### 6.7 Renaming Columns

```python
df.rename(columns={
    'approx_cost(for two people)': 'Cost2plates',
    'listed_in(type)': 'Type'
}, inplace=True)
```

> Clean column names make code more readable and reduce typos.

---

### 6.8 Cleaning the `Cost2plates` Column

**Problem:** Values like `"1,200"` are strings with commas.

```python
def handlecomma(value):
    value = str(value)
    if ',' in value:
        value = value.replace(',', '')
    return float(value)

df['Cost2plates'] = df['Cost2plates'].apply(handlecomma)
```

---

### 6.9 Clustering Rare Categories (Binning)

> **Binning** (also called **bucketing** or **discretization**) groups values into buckets to reduce cardinality and noise.

```python
rest_types = df['rest_type'].value_counts()
rest_types_lessthan1000 = rest_types[rest_types < 1000]

def handle_rest_type(value):
    if value in rest_types_lessthan1000:
        return 'others'
    return value

df['rest_type'] = df['rest_type'].apply(handle_rest_type)
```

| Column | Threshold | Result |
|--------|-----------|--------|
| `rest_type` | < 1000 occurrences → `"others"` | 93 types → 9 types |
| `location` | < 300 occurrences → `"others"` | 93 locations → 42 |
| `cuisines` | < 100 occurrences → `"others"` | 2,704 types → 70 |

> **Why?** Rare categories create visual clutter in charts and statistical noise in models.

---

## 7. Data Visualization Concepts

### 7.1 Count Plot

> Displays the **frequency** of each category as a bar chart.

```python
sns.countplot(x='location', data=df)
plt.xticks(rotation=90)
plt.show()
```

**Use case:** "How many restaurants are in each neighborhood?"

---

### 7.2 Box Plot

> Displays the **statistical distribution** of a numeric variable, grouped by a category.

```python
sns.boxplot(x='online_order', y='rate', data=df)
```

**Anatomy of a Box Plot:**
```
         ┌───────────┐
─── Min  │ Q1  │  Q3 │  Max ───
         │   Median  │
         └───────────┘
              ● ← Outlier
```

| Component | Meaning |
|-----------|---------|
| Bottom of box | Q1 (25th percentile) |
| Line inside box | Median (Q2, 50th percentile) |
| Top of box | Q3 (75th percentile) |
| Box height | IQR (Q3 - Q1) |
| Whisker length | 1.5 × IQR |
| Dots beyond whiskers | Outliers |

**Use case:** "Do restaurants offering online ordering tend to have higher ratings?"

---

### 7.3 Bar Plot / Grouped Bar Plot

> Shows the **aggregate** (mean, sum, count) of a numeric variable per category.

```python
# Total votes per listing type
df.groupby('Type')['votes'].sum().plot(kind='bar')

# Mean rating per restaurant type (Seaborn)
sns.barplot(x='rest_type', y='rate', data=df)
```

**Use case:** "Which listing type gets the most customer votes?"

---

### 7.4 Heatmap

> Visualizes a **matrix of values** using color intensity — great for spotting cross-category patterns.

```python
pivot = df.pivot_table(index='location', columns='online_order', values='rate', aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f')
```

**Use case:** "How does average rating vary by location AND online ordering status simultaneously?"

Color maps:
| `cmap` | Colors |
|--------|--------|
| `'YlGnBu'` | Yellow → Green → Blue |
| `'coolwarm'` | Blue (low) → Red (high) |
| `'viridis'` | Purple → Yellow |
| `'RdYlGn'` | Red → Yellow → Green |

---

### 7.5 Pivot Table

> Reshapes data to show aggregated values across two categorical dimensions.

```python
df.pivot_table(
    index='location',        # Rows
    columns='online_order',  # Columns
    values='rate',           # Values to aggregate
    aggfunc='mean'           # How to aggregate
)
```

Output looks like:

```
online_order     No     Yes
location
Banashankari    3.6    3.9
BTM             3.7    4.0
HSR             3.8    4.1
...
```

> Think of it as a cross-tabulation (cross-tab) — the same concept as an Excel PivotTable.

---

## 8. Key Python Concepts Used

### 8.1 Custom Functions (`def`)

```python
def handlerate(value):
    if value == 'NEW' or value == '-':
        return np.nan
    else:
        value = str(value).split('/')
        return float(value[0])
```

> `def` creates a **reusable function**. The function encapsulates logic and can be called many times without repeating code.

---

### 8.2 `apply()`

```python
df['rate'] = df['rate'].apply(handlerate)
```

> **`Series.apply(func)`** calls a function on every element of a column. It is the Pandas equivalent of a `for` loop over column values, but faster and cleaner.

---

### 8.3 `value_counts()`

```python
df['rest_type'].value_counts()
```

> Returns each unique value and its count, sorted descending. It is the fastest way to understand the **distribution of a categorical column**.

---

### 8.4 Boolean Indexing / Filtering

```python
rare_types = rest_types[rest_types < 1000]
```

> A **boolean condition** creates a Series of `True/False` values. Applying it in `[]` keeps only rows where the condition is `True`. This is the backbone of data filtering in Pandas.

```python
# Example: Filter only high-rated restaurants
high_rated = df[df['rate'] > 4.0]

# Multiple conditions (use & for AND, | for OR)
premium = df[(df['rate'] > 4.0) & (df['Cost2plates'] > 1000)]
```

---

### 8.5 `groupby()`

```python
df.groupby('location')['rate'].mean()
```

> **`groupby()`** is the Split-Apply-Combine pattern:
> 1. **Split** the data into groups (by `location`)
> 2. **Apply** a function (`mean`)
> 3. **Combine** results back into a Series

Common aggregations:
```python
df.groupby('location')['rate'].mean()    # Average rating per location
df.groupby('location')['votes'].sum()   # Total votes per location
df.groupby('location')['name'].count()  # Number of restaurants per location
```

---

## 9. ML Concepts: What Comes Next?

This project is EDA. If you want to build a **predictive model** (e.g., "predict if a restaurant will be highly rated"), here's what you need:

### 9.1 One-Hot Encoding (Deep Dive)

> **The Problem:** ML models cannot understand text labels like `"Yes"`, `"No"`, `"Casual Dining"`.
> **The Solution:** Convert them to numbers using One-Hot Encoding.

#### Step-by-step example for this dataset:

**Input:**
```
online_order | book_table | rest_type      | rate
-------------|------------|----------------|-----
Yes          | No         | Casual Dining  | 4.1
No           | Yes        | Quick Bites    | 3.7
Yes          | Yes        | Cafe           | 4.0
```

**After One-Hot Encoding:**
```
oo_Yes | oo_No | bt_Yes | bt_No | rt_Casual | rt_Quick | rt_Cafe | rate
-------|-------|--------|-------|-----------|----------|---------|-----
1      | 0     | 0      | 1     | 1         | 0        | 0       | 4.1
0      | 1     | 1      | 0     | 0         | 1        | 0       | 3.7
1      | 0     | 1      | 0     | 0         | 0        | 1       | 4.0
```

Now every value is `0` or `1` — the model can do math on this.

#### Python implementation:
```python
# Method 1: Pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['online_order', 'book_table', 'rest_type'], drop_first=True)

# Method 2: Scikit-learn (for pipelines/ML)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_cols = ['online_order', 'book_table', 'rest_type', 'location']
ct = ColumnTransformer([('ohe', OneHotEncoder(drop='first'), categorical_cols)], remainder='passthrough')
X_encoded = ct.fit_transform(df)
```

#### Key parameters:
| Parameter | Effect |
|-----------|--------|
| `drop_first=True` | Drops one column per feature to avoid multicollinearity |
| `sparse=False` | Returns dense array (easier to work with) |
| `handle_unknown='ignore'` | For test data with unseen categories |

---

### 9.2 Train/Test Split

> Before training any model, split your data so that the model is evaluated on **data it has never seen**.

```python
from sklearn.model_selection import train_test_split

X = df.drop('rate', axis=1)   # Features (inputs)
y = df['rate']                 # Target (output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80% training, 20% testing
```

| Split | Purpose |
|-------|---------|
| Training set | Model learns from this |
| Test set | Model is evaluated on this (never seen during training) |

---

### 9.3 Supervised vs Unsupervised Learning

| Type | Definition | Example for this project |
|------|------------|--------------------------|
| **Supervised** | Learn from labeled examples (input → known output) | Predict `rate` from restaurant features |
| **Unsupervised** | Find patterns without labels | Cluster restaurants by characteristics |
| **EDA (this project)** | Understand data before modeling | Descriptive statistics + visualizations |

---

## 10. Key Business Insights

| Business Question | Finding |
|------------------|---------|
| **What type of restaurant?** | **Quick Bites** has the most volume; **Casual Dining** has the highest ratings on average |
| **Best location?** | **BTM**, **HSR**, **Koramangala** have the highest restaurant density → high demand areas |
| **What services?** | Restaurants with **online ordering** consistently have higher ratings and more votes |

---

## 11. Full Workflow Summary

```
📥 Raw CSV (51,717 rows × 17 columns)
         │
         ▼
🔍 Inspect: shape, dtypes, info, head
         │
         ▼
🗑️  Drop irrelevant columns → 11 columns
         │
         ▼
🔁 Drop duplicate rows → 51,609 rows
         │
         ▼
🧹 Clean 'rate': parse "4.1/5" → 4.1, "NEW"/"-" → NaN
         │
         ▼
📊 Impute missing 'rate' with column mean
         │
         ▼
💧 Drop rows with remaining NaN
         │
         ▼
✏️  Rename columns for convenience
         │
         ▼
💰 Clean 'Cost2plates': remove commas, cast to float
         │
         ▼
📦 Bin rare categories into 'others' (rest_type, location, cuisines)
         │
         ▼
📈 Visualize (Count Plot, Box Plot, Bar Plot, Heatmap)
         │
         ▼
💡 Extract business insights ✅
         │
         ▼
🤖 [Next step if needed] One-Hot Encode → Train ML Model
```

---

> 💡 **Key Takeaway:** Data Science is not just about models. A huge portion of real-world work — often 60–80% — is **data cleaning, understanding, and visualization**. This project perfectly demonstrates that reality.
