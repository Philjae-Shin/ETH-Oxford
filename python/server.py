from flask import Flask, request, jsonify
from itertools import permutations

app = Flask(__name__)


@app.route('/compute', methods=['POST'])
def compute():
    """
    Expects a JSON payload like {"num_ones": X, "num_zeros": Y}.
    This creates all permutations containing X '1's and Y '0's.
    For each permutation, we compute an example metric (the sum of digits)
    and return all results as JSON.
    """
    data = request.get_json()
    num_ones = data.get('num_a', 0)
    num_zeros = data.get('num_b', 0)

    # 1) Build the initial list of characters, e.g. if num_ones=2 and num_zeros=1, we get ["1", "1", "0"].
    initial_list = ['a'] * num_ones + ['b'] * num_zeros

    # permutations() will generate duplicates if there are repeated '1's or '0's;
    # we use set() to remove duplicates.
    unique_perms = set(permutations(initial_list))

    results = []
    for perm in unique_perms:
        # perm is a tuple like ('1', '0', '1', ...)
        perm_str = ''.join(perm)

        # As a simple calculation, we sum the digits (in this case, the number of '1's).
        sum_of_digits = sum(int(d) for d in perm_str)

        results.append({
            "permutation": perm_str,
            "calculated_value": sum_of_digits
        })

    # Return all results as JSON
    return jsonify({
        "num_ones": num_ones,
        "num_zeros": num_zeros,
        "count_permutations": len(results),
        "all_results": results
    })


@app.route('/')
def index():
    return "Send a POST request to /compute with JSON like { 'num_ones': <int>, 'num_zeros': <int> }"


if __name__ == '__main__':
    # For local testing. In production, you might use gunicorn or another WSGI server.
    app.run(debug=True, host='0.0.0.0', port=5001)
