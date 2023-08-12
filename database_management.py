import sqlite3
import csv
import os

# delete database file if it already exists
if os.path.exists("car_sharing_database"):
    os.remove("car_sharing_database")

#  task 1 create database
connection = sqlite3.connect("car_sharing_database")
cursor = connection.cursor()

try:
    #  task 1 create CarSharing table
    car_sharing_table_creation = """CREATE TABLE CarSharing (
    id INT,
    timestamp TEXT,
    season TEXT,
    holiday TEXT,
    workingday TEXT,
    weather TEXT,
    temp REAL,
    temp_feel REAL,
    humidity REAL,
    windspeed REAL,
    demand REAL
    )"""

    connection.execute(car_sharing_table_creation)
    print("CarSharing table created successfully")
except:
    print("CarSharing table already exists")

print()

try:
    #   task 1 import csv file
    clear_car_sharing_table = "DELETE FROM CarSharing"
    cursor.execute(clear_car_sharing_table)
    csv_file = open("CarSharing.csv")
    csv_file_rows = csv.reader(csv_file)
    car_sharing_insert_query = """INSERT INTO CarSharing (id, timestamp, season, holiday, workingday, weather, temp, temp_feel, 
        humidity, windspeed, demand) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    cursor.executemany(car_sharing_insert_query, csv_file_rows)

    print("CSV file imported to CarSharing table")
except:
    print("CarSharing table already populated")

print()

try:
    #   task 1 create backup table
    car_sharing_backup = "CREATE TABLE CarSharingBackup AS SELECT * FROM CarSharing"
    cursor.execute(car_sharing_backup)
    print("Backup table created successfully")
except:
    print("Backup table already exists")

try:
    #   task 2 add temperature category column
    temperature_category_alter = "ALTER TABLE CarSharing ADD COLUMN temp_category TEXT"
    cursor.execute(temperature_category_alter)
    print("Temperature category column added successfully")
except:
    print("Temperature category column already exists")

print()

try:
    #   task 2 update temperature category based on temperature feel like value
    temperature_category_update = """
    UPDATE CarSharing SET temp_category = "Cold" WHERE temp_feel < 10;
    UPDATE CarSharing SET temp_category = "Mild" WHERE temp_feel >= 10 AND temp_feel <= 25;
    UPDATE CarSharing SET temp_category = "Hot" WHERE temp_feel > 25;
    """
    cursor.executescript(temperature_category_update)
    print("Temperature category updated successfully")
except:
    print("Temperature category already updated")

print()

try:
    #   task 3 create temperature table
    temperature_creation = "CREATE TABLE Temperature AS SELECT temp, temp_feel, temp_category FROM CarSharing"
    cursor.execute(temperature_creation)
    print("Temperature table created successfully")
except:
    print("Temperature table already exists")

print()

try:
    #   task 3 drop temperature and temperature feel like columns
    temperature_drop = """
    ALTER TABLE CarSharing DROP COLUMN temp;
    ALTER TABLE CarSharing DROP COLUMN temp_feel;
    """
    cursor.executescript(temperature_drop)
    print("Temperature and temperature feel like columns dropped successfully")
except:
    print("Temperature and temperature feel like columns already dropped")

print()

try:
    #   task 4 find distinct values of weather column
    distinct_weather_query = "SELECT DISTINCT weather FROM CarSharing"
    weathers = cursor.execute(distinct_weather_query).fetchall()
    print("Distinct values of weather column are:")
    for distinct_weather in weathers:
        weather = distinct_weather[0]
        print(weather)
except:
    print("An error occurred while finding distinct values of weather column")

print()

try:
    #   task 4 add weather code column
    weather_code_alter = "ALTER TABLE CarSharing ADD COLUMN weather_code INT"
    cursor.execute(weather_code_alter)
    print("Weather code column added successfully")
except:
    print("Weather code column already exists")

print()

try:
    #   task 4 update values of weather code column
    weather_code_update = """
    UPDATE CarSharing SET weather_code = 1 WHERE weather LIKE "%cloudy%";
    UPDATE CarSharing SET weather_code = 2 WHERE weather LIKE "%mist%";
    UPDATE CarSharing SET weather_code = 3 WHERE weather LIKE "%light%";
    UPDATE CarSharing SET weather_code = 4 WHERE weather LIKE "%heavy%";
    """
    cursor.executescript(weather_code_update)
    print("Weather code column updated successfully")
except:
    print("Weather code column already updated")

print()

try:
    #   task 5 create weather table
    weather_creation = "CREATE TABLE Weather AS SELECT weather, weather_code FROM CarSharing"
    cursor.execute(weather_creation)
    print("Weather table created successfully")
except:
    print("Weather table already exists")

print()

try:
    #   task 5 drop weather column
    weather_drop = "ALTER TABLE CarSharing DROP COLUMN weather"
    cursor.execute(weather_drop)
    print("Weather column dropped successfully")
except:
    print("Weather column already dropped")

print()

try:
    #   task 6 create time table
    time_creation = """
    CREATE TABLE Time AS
    SELECT timestamp,
           strftime("%H", timestamp) AS hour,
           CASE cast(strftime('%w', timestamp) as INT)
               WHEN 0 THEN 'Sunday'
               WHEN 1 THEN 'Monday'
               WHEN 2 THEN 'Tuesday'
               WHEN 3 THEN 'Wednesday'
               WHEN 4 THEN 'Thursday'
               WHEN 5 THEN 'Friday'
               WHEN 6 THEN 'Saturday'
           END AS weekday,
           CASE cast(strftime('%m', timestamp) as INT)
               WHEN 1 THEN 'January'
               WHEN 2 THEN 'February'
               WHEN 3 THEN 'March'
               WHEN 4 THEN 'April'
               WHEN 5 THEN 'May'
               WHEN 6 THEN 'June'
               WHEN 7 THEN 'July'
               WHEN 8 THEN 'August'
               WHEN 9 THEN 'September'
               WHEN 10 THEN 'October'
               WHEN 11 THEN 'November'
               WHEN 12 THEN 'December'
           END AS month
        FROM CarSharing
    """
    cursor.execute(time_creation)
    print("Time table created successfully")
except:
    print("Time table already exists")

print()

try:
    #   task 7a select date and time with highest demand rate in 2017
    highest_demand_rate_query = """
    SELECT timestamp, demand FROM CarSharing
    WHERE timestamp LIKE "2017%"
    ORDER BY demand DESC
    LIMIT 1
    """
    result = cursor.execute(highest_demand_rate_query)
    timestamp, demand = result.fetchone()
    print(f"Date and time with highest demand rate in 2017 was {timestamp} with demand rate of {demand}")
except:
    print("An error occurred while finding date and time with highest demand rate in 2017")

print()

#   task 7b select weekday with the highest and lowest average demand rates in 2017
try:
    demand_rate_weekday_query = """
    SELECT DISTINCT weekday, AVG(demand) AS average_demand_rate FROM CarSharing c
    INNER JOIN Time t ON c.timestamp = t.timestamp
    WHERE t.timestamp LIKE "2017%"
    GROUP BY weekday
    ORDER BY average_demand_rate DESC
    """
    result = cursor.execute(demand_rate_weekday_query)
    rows = result.fetchall()
    highest_weekday, highest_average_demand = rows[0]
    lowest_weekday, lowest_average_demand = rows[-1]
    print(f"Weekday with highest average demand rate in 2017 was {highest_weekday} with average demand rate of {highest_average_demand}")
    print(f"Weekday with lowest average demand rate in 2017 was {lowest_weekday} with average demand rate of {lowest_average_demand}")
except Exception as e:
    print(e)
    print("An error occurred while finding weekday with the highest and lowest average demand rates in 2017")

print()

#   task 7b select month with the highest and lowest average demand rates in 2017
try:
    demand_rate_month_query = """
    SELECT DISTINCT month, AVG(demand) AS average_demand_rate FROM CarSharing c
    INNER JOIN Time t ON c.timestamp = t.timestamp
    WHERE t.timestamp LIKE "2017%"
    GROUP BY month
    ORDER BY average_demand_rate DESC
    """
    result = cursor.execute(demand_rate_month_query)
    rows = result.fetchall()
    highest_month, highest_average_demand = rows[0]
    lowest_month, lowest_average_demand = rows[-1]
    print(f"Month with highest average demand rate in 2017 was {highest_month} with average demand rate of {highest_average_demand}")
    print(f"Month with lowest average demand rate in 2017 was {lowest_month} with average demand rate of {lowest_average_demand}")
except Exception as e:
    print(e)
    print("An error occurred while finding month with the highest and lowest average demand rates in 2017")

print()

#   task 7b select season with the highest and lowest average demand rates in 2017
try:
    demand_rate_season_query = """
   SELECT DISTINCT season, AVG(demand) AS average_demand_rate FROM CarSharing
    WHERE timestamp LIKE "2017%"
    GROUP BY season
    ORDER BY average_demand_rate DESC
    """
    result = cursor.execute(demand_rate_season_query)
    rows = result.fetchall()
    highest_season, highest_average_demand = rows[0]
    lowest_season, lowest_average_demand = rows[-1]
    print(f"Season with highest average demand rate in 2017 was {highest_season} with average demand rate of {highest_average_demand}")
    print(f"Season with lowest average demand rate in 2017 was {lowest_season} with average demand rate of {lowest_average_demand}")
except Exception as e:
    print(e)
    print("An error occurred while finding season with the highest and lowest average demand rates in 2017")

print()

#   task 7c select hours of weekday with highest average demand rate in 2017
try:
    demand_rate_hours_query = f"""
    SELECT DISTINCT hour, AVG(demand) AS average_demand_rate FROM CarSharing c
    INNER JOIN Time t ON c.timestamp = t.timestamp
    WHERE weekday = "{highest_weekday}" AND t.timestamp LIKE "2017%"
    GROUP BY hour
    ORDER BY average_demand_rate DESC    """
    result = cursor.execute(demand_rate_hours_query)
    rows = result.fetchall()

    print(f"Hours of {highest_weekday} with highest average demand rate in 2017 are:")
    for current_row in rows:
        hour, average_demand = current_row
        print(f"Hour: {hour} | Average demand rate: {average_demand}")

except Exception as e:
    print(e)
    print("An error occurred while finding hours of weekday with highest average demand rate in 2017")

print()

#   task 7d select most frequent temperature in 2017
try:
    most_frequent_temperature_query = """
    SELECT DISTINCT temp_category most_frequent_temperature FROM CarSharing
    WHERE timestamp LIKE "2017%"
    GROUP BY temp_category
    ORDER BY count(*) DESC
    LIMIT 1
    """
    result = cursor.execute(most_frequent_temperature_query)
    most_frequent_temperature = result.fetchone()[0]
    print(f"Most frequent temperature in 2017 was {most_frequent_temperature}")
except:
    print("An error occurred while finding most frequent temperature in 2017")

print()

#   task 7d select most frequent weather in 2017
try:
    most_frequent_weather_query = """
    SELECT DISTINCT w.weather most_frequent_weather FROM CarSharing c
    INNER JOIN Weather w on c.weather_code = w.weather_code
    WHERE timestamp LIKE "2017%"
    GROUP BY c.weather_code
    ORDER BY count(c.weather_code) DESC
    LIMIT 1
    """
    result = cursor.execute(most_frequent_weather_query)
    most_frequent_weather = result.fetchone()[0]
    print(f"Most frequent weather in 2017 was {most_frequent_weather}")
except Exception as e:
    print(e)
    print("An error occurred while finding most frequent weather in 2017")

print()

#   task 7d select average windspeed by months in 2017
try:
    average_windspeed_query = """
    SELECT month, AVG(windspeed) average_windspeed FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%"
    GROUP BY month
    ORDER BY c.timestamp
    """
    result = cursor.execute(average_windspeed_query)
    rows = result.fetchall()

    print("Average windspeed by months in 2017:")

    for current_row in rows:
        month, average_windspeed = current_row
        print(f"Month: {month} | Average windspeed: {average_windspeed}")
except Exception as e:
    print(e)
    print("An error occurred while finding average windspeed by months in 2017")

print()

#   task 7d select highest and lowest windspeeds in 2017
try:
    windspeed_query = """
   SELECT month, windspeed FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%"
    GROUP BY month
    ORDER BY windspeed DESC
    """
    result = cursor.execute(windspeed_query)
    rows = result.fetchall()
    highest_windspeed_month, highest_windspeed = rows[0]
    lowest_windspeed_month, lowest_windspeed = rows[-1]
    print(f"Month with highest windspeed in 2017 was {highest_windspeed_month} with windspeed of {highest_windspeed}")
    print(f"Month with lowest windspeed in 2017 was {lowest_windspeed_month} with windspeed of {lowest_windspeed}")
except Exception as e:
    print(e)
    print("An error occurred while finding highest and lowest windspeeds in 2017")

print()

#   task 7d select average humidity by months in 2017
try:
    average_humidity_query = """
    SELECT month, AVG(humidity) average_humidity FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%"
    GROUP BY month
    ORDER BY c.timestamp
    """
    result = cursor.execute(average_humidity_query)
    rows = result.fetchall()

    print("Average humidity by months in 2017:")

    for current_row in rows:
        month, average_humidity = current_row
        print(f"Month: {month} | Average humidity: {average_humidity}")
except Exception as e:
    print(e)
    print("An error occurred while finding average humidity by months in 2017")

print()

#   task 7d select highest and lowest humidities in 2017
try:
    humidity_query = """
   SELECT month, humidity FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%"
    GROUP BY month
    ORDER BY humidity DESC
    """
    result = cursor.execute(humidity_query)
    rows = result.fetchall()
    highest_humidity_month, highest_humidity = rows[0]
    lowest_humidity_month, lowest_humidity = rows[-1]
    print(f"Month with highest humidity in 2017 was {highest_humidity_month} with humidity of {highest_humidity}")
    print(f"Month with lowest humidity in 2017 was {lowest_humidity_month} with humidity of {lowest_humidity}")
except Exception as e:
    print(e)
    print("An error occurred while finding highest and lowest humidity in 2017")

print()

#   task 7d select average demand rate by temperature in 2017
try:
    average_demand_rate_by_temperature_query = """
    SELECT temp_category, AVG(demand) average_demand_rate FROM CarSharing c
    WHERE timestamp LIKE "2017%"
    GROUP BY temp_category
    ORDER BY average_demand_rate DESC
    """
    result = cursor.execute(average_demand_rate_by_temperature_query)
    rows = result.fetchall()

    print("Average demand rates by temperature in 2017:")

    for current_row in rows:
        temperature, average_demand = current_row
        print(f"Temperature: {temperature} | Average demand rate: {average_demand}")
except Exception as e:
    print(e)
    print("An error occurred while finding average demand rates by temperature in 2017")

print()

#   task 7e select most frequent temperature in month with highest demand rate in 2017
try:
    most_frequent_temperature_query = f"""
    SELECT DISTINCT temp_category most_frequent_temperature FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%" AND month = "{highest_month}"
    GROUP BY temp_category
    ORDER BY count(*) DESC
    LIMIT 1
    """
    result = cursor.execute(most_frequent_temperature_query)
    most_frequent_temperature = result.fetchone()[0]
    print(f"Most frequent temperature in {highest_month}, 2017 was {most_frequent_temperature}")
except:
    print(f"An error occurred while finding most frequent temperature in {highest_month}, 2017")

print()

#   task 7e select most frequent weather in month with highest demand rate in 2017
try:
    most_frequent_weather_query = f"""
    SELECT DISTINCT w.weather most_frequent_weather FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    INNER JOIN Weather w on c.weather_code = w.weather_code
    WHERE c.timestamp LIKE "2017%" AND month = "{highest_month}"
    GROUP BY c.weather_code
    ORDER BY count(c.weather_code) DESC
    LIMIT 1
    """
    result = cursor.execute(most_frequent_weather_query)
    most_frequent_weather = result.fetchone()[0]
    print(f"Most frequent weather in {highest_month}, 2017 was {most_frequent_weather}")
except Exception as e:
    print(e)
    print(f"An error occurred while finding most frequent weather in {highest_month}, 2017")

print()

#   task 7e select average windspeed in month with highest demand rate in 2017
try:
    average_windspeed_query = f"""
    SELECT month, AVG(windspeed) average_windspeed FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%" AND month = "{highest_month}"
    GROUP BY month    """
    result = cursor.execute(average_windspeed_query)
    rows = result.fetchall()

    print(f"Average windspeed in {highest_month}, 2017:")

    for current_row in rows:
        month, average_windspeed = current_row
        print(f"Month: {month} | Average windspeed: {average_windspeed}")
except Exception as e:
    print(e)
    print(f"An error occurred while finding average windspeed in {highest_month}, 2017")

print()

#   task 7e select highest and lowest windspeeds in month with highest demand rate in 2017
try:
    windspeed_query = f"""
   SELECT windspeed FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%" AND month = "{highest_month}" AND length(windspeed) > 0
    ORDER BY windspeed DESC
    """
    result = cursor.execute(windspeed_query)
    rows = result.fetchall()
    highest_windspeed = rows[0][0]
    lowest_windspeed = rows[-1][0]
    print(f"Highest windspeed in {highest_month}, 2017 was {highest_windspeed}")
    print(f"Lowest windspeed in {highest_month}, 2017 was {lowest_windspeed}")
except Exception as e:
    print(e)
    print(f"An error occurred while finding highest and lowest windspeeds in {highest_month}, 2017")

print()

#   task 7e select average humidity in month with highest demand rate in 2017
try:
    average_humidity_query = f"""
    SELECT month, AVG(humidity) average_humidity FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%" AND month = "{highest_month}"
    GROUP BY month
    """
    result = cursor.execute(average_humidity_query)
    rows = result.fetchall()

    print(f"Average humidity in {highest_month}, 2017:")

    for current_row in rows:
        month, average_humidity = current_row
        print(f"Month: {month} | Average humidity: {average_humidity}")
except Exception as e:
    print(e)
    print(f"An error occurred while finding average humidity in {highest_month}, 2017")

print()

#   task 7e select highest and lowest humidities in month with highest demand rate in 2017
try:
    humidity_query = f"""
    SELECT  humidity FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%" AND month = "{highest_month}" AND length(humidity) > 0
    ORDER BY humidity DESC    
    """
    result = cursor.execute(humidity_query)
    rows = result.fetchall()
    highest_humidity = rows[0][0]
    lowest_humidity = rows[-1][0]
    print(f"Highest humidity in 2017 was {highest_humidity}")
    print(f"Lowest humidity in 2017 was {lowest_humidity}")

except Exception as e:
    print(e)
    print(f"An error occurred while finding highest and lowest windspeeds in {highest_month}, 2017")

print()

#   task 7e select average demand rate by temperature in month with highest demand rate in 2017
try:
    average_demand_rate_by_temperature_query = f"""
    SELECT temp_category, AVG(demand) average_demand_rate FROM CarSharing c
    INNER JOIN Time t on c.timestamp = t.timestamp
    WHERE c.timestamp LIKE "2017%" AND month = "{highest_month}"
    GROUP BY temp_category
    ORDER BY average_demand_rate DESC
    """
    result = cursor.execute(average_demand_rate_by_temperature_query)
    rows = result.fetchall()

    print(f"Average demand rates by temperature in {highest_month}, 2017:")

    for current_row in rows:
        temperature, average_demand = current_row
        print(f"Temperature: {temperature} | Average demand rate: {average_demand}")
except Exception as e:
    print(e)
    print(f"An error occurred while finding average demand rates by temperature in {highest_month}, 2017")

print()

connection.commit()
connection.close()

print("Database management tasks have successfully been completed")