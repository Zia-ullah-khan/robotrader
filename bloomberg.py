import requests
def get_bloomberg_data(url):
    """Fetch data from Bloomberg API."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from Bloomberg: {e}")
        return None
data = get_bloomberg_data("https://www.bloomberg.com/search?query=Apple")
print(data)