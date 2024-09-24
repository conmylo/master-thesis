import datetime

def get_time_of_day(unix_time):
    # Convert unix timestamp to datetime
    dt = datetime.datetime.utcfromtimestamp(unix_time)

    # Extract the hour from the datetime object
    hour = dt.hour

    # Categorize the time of day
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

# Example usage
unix_time = 169545394  # Example Unix time
time_of_day = get_time_of_day(unix_time)
print(time_of_day)
