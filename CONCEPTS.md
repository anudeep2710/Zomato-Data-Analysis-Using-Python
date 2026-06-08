# 📊 Zomato Data Analysis — Concept Guide

A beginner-friendly, in-depth explanation of every concept, library, function, and technique used in the **Zomato Dataset Analysis** Jupyter Notebook.

---

## 📚 Table of Contents

1. [Project Goal](#1-project-goal)
2. [Libraries Used](#2-libraries-used)
3. [Dataset Overview](#3-dataset-overview)
4. [Data Pre-Processing Concepts](#4-data-pre-processing-concepts)
   - [Reading a CSV File](#41-reading-a-csv-file)
   - [Inspecting the DataFrame](#42-inspecting-the-dataframe)
   - [Dropping Columns](#43-dropping-columns)
   - [Dropping Duplicates](#44-dropping-duplicates)
   - [Cleaning the `rate` Column](#45-cleaning-the-rate-column)
   - [Handling Null Values](#46-handling-null-values)
   - [Renaming Columns](#47-renaming-columns)
   - [Cleaning the `Cost2plates` Column](#48-cleaning-the-cost2plates-column)
   - [Clustering / Grouping Rare Categories](#49-clustering--grouping-rare-categories)
5. [Data Visualization Concepts](#5-data-visualization-concepts)
   - [Count Plot](#51-count-plot)
   - [Box Plot](#52-box-plot)
   - [Bar Plot / Grouped Bar Plot](#53-bar-plot--grouped-bar-plot)
   - [Heatmap](#54-heatmap)
6. [Key Business Insights](#6-key-business-insights)
7. [Python Concepts Used](#7-python-concepts-used)
   - [Custom Functions (def)](#71-custom-functions-def)
   - [apply()](#72-apply)
   - [value_counts()](#73-value_counts)
   - [Boolean Indexing / Filtering](#74-boolean-indexing--filtering)
8. [Summary of the Full Workflow](#8-summary-of-the-full-workflow)

---

## 1. Project Goal

A client wants to open a new restaurant and needs data-driven answers to three key questions:

| # | Question |
|---|----------|
| 1 | What **type** of restaurant should be opened? |
| 2 | What is the **best location** to open it? |
| 3 | What **services** (e.g., online order, table booking) should it offer? |

The dataset comes from **Kaggle** and contains real Zomato restaurant listings from **Bangalore, India**.

---

## 2. Libraries Used

### 🐼 Pandas (`import pandas as pd`)
> **What it is:** The most popular Python library for data manipulation and analysis.

Pandas introduces the concept of a **DataFrame** — a table-like structure (rows and columns) similar to an Excel spreadsheet. It lets you:
- Load data from CSV files
- Filter, sort, and transform data
- Handle missing values
- Rename and drop columns

```python
import pandas as pd
```

---

### 🔢 NumPy (`import numpy as np`)
> **What it is:** A library for numerical computing in Python.

NumPy is used here primarily for one thing: `np.nan`, which represents a **"Not a Number"** value — a standard way to represent missing data.

```python
import numpy as np
```

---

### 📈 Matplotlib (`import matplotlib.pyplot as plt`)
> **What it is:** The foundational plotting library in Python.

Matplotlib provides the canvas and axes for all plots. Even when Seaborn is used, it internally relies on Matplotlib.

```python
import matplotlib.pyplot as plt
plt.style.use('dark_background')  # Sets a dark theme for all plots
```

---

### 🎨 Seaborn (`import seaborn as sns`)
> **What it is:** A high-level data visualization library built on top of Matplotlib.

Seaborn makes it easy to create statistical charts like:
- Count plots
- Box plots
- Bar plots
- Heatmaps

```python
import seaborn as sns
```

---

## 3. Dataset Overview

The raw dataset (`zomato.csv`) contains **51,717 rows** and **17 columns**:

| Column | Description |
|--------|-------------|
| `url` | Link to the restaurant's Zomato page |
| `address` | Full address |
| `name` | Restaurant name |
| `online_order` | Whether online ordering is available (`Yes`/`No`) |
| `book_table` | Whether table booking is available (`Yes`/`No`) |
| `rate` | Rating out of 5 (e.g., `4.1/5`) |
| `votes` | Number of votes/reviews |
| `phone` | Phone number |
| `location` | Neighborhood/area |
| `rest_type` | Restaurant type (e.g., Casual Dining, Cafe) |
| `dish_liked` | Popular dishes |
| `cuisines` | Types of cuisines served |
| `approx_cost(for two people)` | Estimated cost for two people |
| `reviews_list` | List of reviews |
| `menu_item` | Menu items |
| `listed_in(type)` | Listing category (e.g., Buffet, Delivery) |
| `listed_in(city)` | City/area it's listed under |

---

## 4. Data Pre-Processing Concepts

Data rarely comes clean. This section covers every cleaning step performed on the dataset.

---

### 4.1 Reading a CSV File

```python
df = pd.read_csv('zomato.csv')
df.head()
```

> **`pd.read_csv()`** loads a CSV file into a Pandas DataFrame.
>
> **`df.head()`** displays the first 5 rows — useful for a quick visual check.

---

### 4.2 Inspecting the DataFrame

```python
df.shape       # (51717, 17) → 51,717 rows and 17 columns
df.columns     # Lists all column names
df.info()      # Shows data types and count of non-null values per column
```

> **`df.shape`** returns a tuple `(rows, columns)`.
>
> **`df.columns`** lists all column names as an Index object.
>
> **`df.info()`** prints a concise summary: column names, non-null counts, and data types. This is the first step to spotting missing data.

---

### 4.3 Dropping Columns

```python
df = df.drop(['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list'], axis=1)
```

> **`df.drop(columns, axis=1)`** removes specified columns.
>
> `axis=1` means "operate on columns" (as opposed to `axis=0` which operates on rows).
>
> **Why?** Columns like `url`, `address`, `phone`, `menu_item`, `dish_liked`, and `reviews_list` are not useful for this analysis. Removing them simplifies the dataset from 17 columns to 11.

---

### 4.4 Dropping Duplicates

```python
df.drop_duplicates(inplace=True)
df.shape  # (51609, 11)
```

> **`df.drop_duplicates()`** removes rows that are exact copies of each other.
>
> `inplace=True` modifies the original DataFrame directly (instead of returning a new one).
>
> **Why?** Duplicate rows can skew analysis results. After dropping, we went from 51,717 to 51,609 rows.

---

### 4.5 Cleaning the `rate` Column

**Problem:** The `rate` column has values like `"4.1/5"`, `"NEW"`, `"-"`, and even `NaN` — all in string format.

**Solution:** Extract just the numeric part.

```python
def handlerate(value):
    if value == 'NEW' or value == '-':
        return np.nan          # Treat 'NEW' and '-' as missing values
    else:
        value = str(value).split('/')
        value = value[0]       # Take only "4.1" from "4.1/5"
        return float(value)    # Convert to a number

df['rate'] = df['rate'].apply(handlerate)
```

> **`str.split('/')`** splits the string on `/`, returning a list: `["4.1", "5"]`.
>
> **`float()`** converts a string number like `"4.1"` to a floating-point number `4.1`.
>
> **`np.nan`** is used as the standard representation for missing numeric values in Pandas.

---

### 4.6 Handling Null Values

After cleaning the `rate` column, there are still **10,019 null values** in it.

**Strategy 1: Fill with the mean**

```python
df['rate'].fillna(df['rate'].mean(), inplace=True)
df['rate'].isnull().sum()  # 0 — no more nulls!
```

> **`df['rate'].mean()`** calculates the average of all non-null values.
>
> **`fillna(value)`** replaces all `NaN` values with the given value.
>
> **Why mean?** It's a common, neutral imputation strategy for numeric data when the distribution is roughly normal.

**Strategy 2: Drop remaining nulls** (for other columns like `location`, `cuisines`, `rest_type`)

```python
df.dropna(inplace=True)
```

> **`df.dropna()`** removes any row that contains at least one `NaN` value.
>
> **Why?** The null counts in these columns were small enough (< 1%) that dropping them doesn't significantly affect the dataset.

---

### 4.7 Renaming Columns

```python
df.rename(columns={'approx_cost(for two people)': 'Cost2plates', 'listed_in(type)': 'Type'}, inplace=True)
```

> **`df.rename(columns={old: new})`** renames specified columns.
>
> **Why?** Column names like `approx_cost(for two people)` are awkward to type. Shorter names like `Cost2plates` are more convenient for analysis.

---

### 4.8 Cleaning the `Cost2plates` Column

**Problem:** Values like `"1,200"` contain commas and are stored as strings.

**Solution:** Remove commas and convert to float.

```python
def handlecomma(value):
    value = str(value)
    if ',' in value:
        value = value.replace(',', '')  # Remove comma: "1,200" → "1200"
    return float(value)                 # Convert to number

df['Cost2plates'] = df['Cost2plates'].apply(handlecomma)
```

> **`str.replace(',', '')`** removes all commas from the string.
>
> **`float()`** converts to a numeric type, enabling arithmetic operations.

---

### 4.9 Clustering / Grouping Rare Categories

Some columns like `rest_type`, `location`, and `cuisines` have many unique values, but most of them are very rare. Grouping rare values into an "others" category makes visualization cleaner.

**Example: `rest_type` column**

```python
rest_types = df['rest_type'].value_counts(ascending=False)
rest_types_lessthan1000 = rest_types[rest_types < 1000]

def handle_rest_type(value):
    if value in rest_types_lessthan1000:
        return 'others'
    else:
        return value

df['rest_type'] = df['rest_type'].apply(handle_rest_type)
```

> **`value_counts()`** counts how many times each unique value appears, sorted by frequency (descending by default).
>
> **Boolean indexing** (`rest_types[rest_types < 1000]`) filters the Series to keep only entries with a count below 1000.
>
> **Why?** Restaurant types that appear fewer than 1,000 times are lumped into a single "others" bucket. This prevents the charts from being cluttered with dozens of tiny bars.

The same pattern is applied to `location` (threshold: 300) and `cuisines` (threshold: 100).

---

## 5. Data Visualization Concepts

### 5.1 Count Plot

> **What it is:** A bar chart that shows the count (frequency) of each category.

```python
sns.countplot(x='location', data=df)
plt.title('Count Plot of Various Locations')
plt.xticks(rotation=90)
plt.show()
```

> **`sns.countplot(x='column', data=df)`** automatically counts rows per category and draws bars.
>
> **`plt.xticks(rotation=90)`** rotates x-axis labels so they don't overlap.
>
> **Use case:** Answers "How many restaurants are in each location?"

---

### 5.2 Box Plot

> **What it is:** Shows the distribution of a numeric variable — including the median, quartiles, and outliers.

```python
sns.boxplot(x='online_order', y='rate', data=df)
plt.title('Online Order vs. Rating')
plt.show()
```

**Reading a box plot:**
```
         ┌─────────────┐
Min ─────┤  Q1    Q3   ├───── Max
         │      Median │
         └─────────────┘
                ●  ← Outlier
```

> **`Q1`** = 25th percentile, **`Q3`** = 75th percentile
>
> The **box** covers the Interquartile Range (IQR = Q3 - Q1)
>
> **Whiskers** extend to 1.5× IQR beyond Q1/Q3
>
> Dots beyond whiskers are **outliers**
>
> **Use case:** Answers "Do restaurants with online ordering tend to have higher ratings?"

---

### 5.3 Bar Plot / Grouped Bar Plot

> **What it is:** Visualizes the aggregate value (e.g., mean) of a numeric variable per category.

```python
df.groupby(['listed_in(type)'])['votes'].sum().plot(kind='bar')
plt.title('Votes by Listing Type')
plt.show()
```

OR with Seaborn:

```python
sns.barplot(x='Type', y='votes', data=df)
```

> **`groupby('column')['col2'].sum()`** groups the DataFrame by a column and computes a sum for another column.
>
> **`plot(kind='bar')`** is Pandas' built-in plotting, which internally uses Matplotlib.
>
> **Use case:** Answers "Which type of listing (Delivery, Dine-out, Buffet...) gets the most votes?"

---

### 5.4 Heatmap

> **What it is:** Displays a matrix of values as a color-encoded grid.

```python
pivot = df.pivot_table(index='location', columns='online_order', values='rate', aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap='YlGnBu')
plt.title('Location vs Online Order vs Avg Rating')
plt.show()
```

> **`pivot_table()`** reshapes data into a 2D summary table (like an Excel pivot table).
>
> **`annot=True`** prints the actual values inside each cell.
>
> **`cmap='YlGnBu'`** sets the color palette (Yellow → Green → Blue for low → high).
>
> **Use case:** Answers "How does the average rating differ by location and whether online ordering is available?"

---

## 6. Key Business Insights

After cleaning and visualization, the notebook draws these actionable insights:

| Question | Insight |
|----------|---------|
| **Type of restaurant?** | **Quick Bites** and **Casual Dining** dominate the market. Casual Dining tends to have higher ratings. |
| **Best location?** | **BTM**, **HSR**, and **Koramangala** have the highest number of restaurants, indicating demand. |
| **Services to provide?** | Most high-rated restaurants offer **online ordering**. Table booking is less common but seen in premium restaurants. |

---

## 7. Python Concepts Used

### 7.1 Custom Functions (`def`)

```python
def handlerate(value):
    if value == 'NEW' or value == '-':
        return np.nan
    else:
        value = str(value).split('/')
        return float(value[0])
```

> **`def`** defines a reusable block of code.
>
> Functions are used here to encapsulate the cleaning logic for each column.

---

### 7.2 `apply()`

```python
df['rate'] = df['rate'].apply(handlerate)
```

> **`Series.apply(func)`** applies a function to every element in a Pandas Series (column).
>
> It is equivalent to a loop, but runs more efficiently and with cleaner code.

---

### 7.3 `value_counts()`

```python
df['rest_type'].value_counts()
```

> Returns a Series of unique values and their counts, sorted from most to least frequent.
>
> Very useful for understanding the distribution of a categorical column.

---

### 7.4 Boolean Indexing / Filtering

```python
rest_types_lessthan1000 = rest_types[rest_types < 1000]
```

> In Pandas, you can filter a Series or DataFrame using a **boolean condition** inside `[]`.
>
> `rest_types < 1000` creates a True/False mask; applying it returns only rows where the condition is True.

---

## 8. Summary of the Full Workflow

```
Raw CSV (51,717 rows × 17 columns)
         ↓
  Drop irrelevant columns (→ 11 columns)
         ↓
  Drop duplicate rows (→ 51,609 rows)
         ↓
  Clean 'rate' column (remove /5, NEW, -)
         ↓
  Fill null ratings with mean
         ↓
  Drop remaining null rows
         ↓
  Rename columns for convenience
         ↓
  Clean 'Cost2plates' (remove commas)
         ↓
  Cluster rare rest_type, location, cuisines → 'others'
         ↓
  Visualize with Count Plots, Box Plots, Bar Plots, Heatmaps
         ↓
  Draw business insights ✅
```

---

> 💡 **Key Takeaway:** This project is a classic **Exploratory Data Analysis (EDA)** workflow. The goal is not to build a predictive model, but to clean messy real-world data and extract meaningful, business-relevant patterns from it.
