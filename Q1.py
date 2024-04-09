import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def DetectAndRemove(df, col_name):
    tr = 3
    mean = df[col_name].mean()
    std = df[col_name].std()

    # Create a mask to filter out outliers
    mask = np.abs((df[col_name] - mean) / std) <= tr

    # Return a new DataFrame without outliers
    cleaned_df = df[mask]
    return cleaned_df


# Read the file
with open('Q1_property.csv', 'r') as file:
    lines = file.readlines()

# Process the lines and split them based on a specific separator
data = [line.strip().split(';') for line in lines]

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Set the first row as column headers
df.columns = df.iloc[0]

# Remove the first row (which is now the header)
df = df[1:]

# --------------------------------------------Filling missing values--------------------------------------------------------------------------------------------

df = pd.read_csv('Q1_propertyCLEANED4.csv', na_values='""')
# print(df.isnull().sum())
df['agency'] = df['agency'].fillna('Unknown')
df['agent'] = df['agent'].fillna('Unknown')
# print(df.isnull().sum())

##--------------------------------------------Renaming Columns--------------------------------------------------------------------------------------------

df = df.rename(columns={'"property_type"': 'property_type'})
df = df.rename(columns={'"property_id"': 'property_id'})
df = df.rename(columns={'"location_id"': 'location_id'})
df = df.rename(columns={'"price"': 'price'})
df = df.rename(columns={'"bedrooms"': 'bedrooms'})
df = df.rename(columns={'"baths"': 'baths'})
df = df.rename(columns={'"latitude"': 'latitude'})
df = df.rename(columns={'"location"': 'location'})
df = df.rename(columns={'"city"': 'city'})
df = df.rename(columns={'"province_name"': 'province_name'})
df = df.rename(columns={'"area"': 'area'})
df = df.rename(columns={'"purpose"': 'purpose'})
df = df.rename(columns={'"page_url"': 'page_url'})
df = df.rename(columns={'"image_urls"': 'image_urls'})
df = df.rename(columns={'"description"': 'description'})
df = df.rename(columns={'"title"': 'title'})
df = df.rename(columns={'"date_added"': 'date_added'})
df = df.rename(columns={'"date_updated"': 'date_updated'})
df = df.rename(columns={'"date_deleted"': 'date_deleted'})
df = df.rename(columns={'"longitude"': 'longitude'})
df = df.rename(columns={'"agency"': 'agency'})
df = df.rename(columns={'"agent"': 'agent'})

##--------------------------------------------Removing "" from every value--------------------------------------------------------------------------------------------

df['agency'] = df['agency'].str.replace('"', '')
df['location'] = df['location'].str.replace('"', '')
df['agent'] = df['agent'].str.replace('"', '')
df['page_url'] = df['page_url'].str.replace('"', '')
df['purpose'] = df['purpose'].str.replace('"', '')
df['area'] = df['area'].str.replace('"', '')
df['city'] = df['city'].str.replace('"', '')
df['province_name'] = df['province_name'].str.replace('"', '')
df['date_added'] = df['date_added'].str.replace('"', '')
df['property_type'] = df['property_type'].str.replace('"', '')

##--------------------------------------------Converting to appropriate dtypes--------------------------------------------------------------------------------------------

df['property_id'] = df['property_id'].astype(int)
df['location_id'] = df['location_id'].astype(int)
df['price'] = df['price'].astype(int)
df['bedrooms'] = df['bedrooms'].astype(int)
df['baths'] = df['baths'].astype(int)
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)
df['location'] = df['location'].astype(str)
df['city'] = df['city'].astype(str)
df['province_name'] = df['province_name'].astype(str)
df['area'] = df['area'].astype(str)
df['purpose'] = df['purpose'].astype(str)
df['page_url'] = df['page_url'].astype(str)

# print(df.dtypes)

##--------------------------------------------Detecting Outliers--------------------------------------------------------------------------------------------

df = DetectAndRemove(df,'price')
df = DetectAndRemove(df,'bedrooms')
df = DetectAndRemove(df,'baths')
df = DetectAndRemove(df,'latitude')
df = DetectAndRemove(df,'longitude')

df.to_csv('New0.csv', index=False)
 
##--------------------------------------------EDA--------------------------------------------------------------------------------------------
    
df.correlation = df[['price', 'latitude', 'longitude', 'baths', 'bedrooms']]

property_counts = df.groupby(['agency', 'agent']).size().reset_index(name='property_count')

avg_prices = df.groupby(['agency', 'agent'])['price'].mean().reset_index(name='avg_price')

corr2 = property_counts['property_count']
corr3 = avg_prices['avg_price']
print(corr3)
merged = pd.concat([corr2, corr3], axis=1)
print(merged.corr())


##--------------------------------------------Additional temporal features--------------------------------------------------------------------------------------------

date_list_dash = [date.replace('/', '-') for date in df['date_added']]

df['date_added'] = date_list_dash
df['date_added'] = pd.to_datetime(df['date_added'], format='%m-%d-%Y', errors='coerce')

df['month'] = df['date_added'].dt.month
df['day'] = df['date_added'].dt.day
df.to_csv('New1.csv', index=False)

df['date_added'].astype('datetime64[ns]')

numerical_columns = [ 'price', 'longitude', 'latitude', 'baths', 'bedrooms']

means = df[numerical_columns].mean()
std_devs = df[numerical_columns].std()

for col in numerical_columns:
    df[col + '_standardized'] = (df[col] - means[col]) / std_devs[col]

df.to_csv('New2.csv', index=False)
df = pd.read_csv('New2.csv')
# print(df.tail())

##--------------------------------------------Feature Engineering--------------------------------------------------------------------------------------------

df['area'] = df['area'].str.replace(',', '')

square_meters_per_marla = 20.9  # Replace this with the appropriate conversion factor
square_meters_per_kanal = 505.857  # Replace this with the appropriate conversion factor

def convert_to_square_meters(area):
    value, unit = area.split()
    value = float(value)
    
    if unit.lower() == 'marla':
        return value * square_meters_per_marla
    elif unit.lower() == 'kanal':
        return value * square_meters_per_kanal
    else:
        return "Unit not recognized"
    
df['price per square meter'] = df['area'].apply(convert_to_square_meters)
df['price per square meter'] = df['price'] / df['price per square meter']
df.to_csv('New3.csv', index=False)
df = pd.read_csv('New3.csv')
# print(df.tail())

##--------------------------------------------Encoding categorical features--------------------------------------------------------------------------------------------

Area = df['area']

df = df.drop(['area'], axis=1)

ordinal_encoder = OrdinalEncoder()

Area_encoded = ordinal_encoder.fit_transform(Area.values.reshape(-1, 1))

df['area'] = Area_encoded

df['area'] = df['area'].astype(int)

# df.to_csv('New5.csv', index=False)

df = df.drop(['property_id'], axis=1)
df = df.drop(['location_id'], axis=1)
df = df.drop(['page_url'], axis=1)
df = df.drop(['date_added'], axis=1)
df = df.drop(['agent'], axis=1)
df = df.drop(['agency'], axis=1)
df = df.drop(['location'], axis=1)
df = df.drop(['city'], axis=1)
df = df.drop(['province_name'], axis=1)

PropertyType = df['property_type']
df = df.drop(['property_type'], axis=1)
OHE = OneHotEncoder()
PropertyType_encoded = OHE.fit_transform(PropertyType.values.reshape(-1, 1))
PropertyType_encoded = pd.DataFrame(PropertyType_encoded.toarray(), columns=OHE.categories_)
PropertyType_encoded = PropertyType_encoded.astype(int)
df = pd.concat([df, PropertyType_encoded], axis=1)

Purpose = df['purpose']
df = df.drop(['purpose'], axis=1)
OHE = OneHotEncoder()
Purpose_encoded = OHE.fit_transform(Purpose.values.reshape(-1, 1))
Purpose_encoded = pd.DataFrame(Purpose_encoded.toarray(), columns=OHE.categories_)
Purpose_encoded = Purpose_encoded.astype(int)
df = pd.concat([df, Purpose_encoded], axis=1)
# df.to_csv('New7(FINAL).csv', index=False)

df.columns = df.columns.astype(str)


##--------------------------------------------Splitting the data--------------------------------------------------------------------------------------------

df = df.drop(['price_standardized'], axis=1)
df = df.drop(['longitude_standardized'], axis=1)
df = df.drop(['latitude_standardized'], axis=1)
df = df.drop(['baths_standardized'], axis=1)
df = df.drop(['bedrooms_standardized'], axis=1)
df = df.drop(['price per square meter'], axis=1)

print(df.dtypes)

X = df.drop(columns=['price'])  # Features
y = df['price']  # Target variable

# Split the dataset into training and test sets (421 rows for training, rest for test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, train_size=0.7, random_state=42)

# Initialize the regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate the root mean squared error
rmse = np.sqrt(mse)

# Calculate the mean absolute percentage error
mape = mean_absolute_percentage_error(y_test, y_pred)

# Print the performance metrics
print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('Mean Absolute Percentage Error:', mape)


















