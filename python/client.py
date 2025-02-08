import sys
import requests
import json

def main():
    """
    Usage:
        python client.py <num_ones> <num_zeros>
    Example:
        python client.py 2 1
    This will send a POST request to http://localhost:5001/compute with JSON { num_ones, num_zeros }
    and then print the response.
    """
    if len(sys.argv) != 3:
        print("Usage: python client.py <num_ones> <num_zeros>")
        sys.exit(1)

    num_ones = int(sys.argv[1])
    num_zeros = int(sys.argv[2])

    url = "http://localhost:5001/compute"
    payload = {
        "num_ones": num_ones,
        "num_zeros": num_zeros
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error if HTTP status is not 200
    except requests.exceptions.RequestException as e:
        print(f"Error during HTTP request: {e}")
        sys.exit(1)

    # Parse JSON response
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Response is not valid JSON.")
        sys.exit(1)

    # Print the result
    print("Server Response:")
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()