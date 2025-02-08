# run_haskell.py
import sys
import subprocess

def main():
    """
    Usage:
      python run_haskell.py A A B A B B A B

    This will run the compiled Haskell program 'PermutationProgram8'
    with the provided 8 arguments.
    """
    if len(sys.argv) != 9:
        print("Please provide exactly 8 arguments (A or B). Example:")
        print("  python run_haskell.py A A B A B B A B")
        sys.exit(1)

    # The first argument is the script name, so extract the next 8 arguments
    cmd = ["../haskell/PermutationProgram8"] + sys.argv[1:]

    try:
        # Run the Haskell executable and capture both stdout and stderr
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print the outputs
        if result.stdout:
            print("=== Haskell Program Output ===")
            print(result.stdout)
        if result.stderr:
            print("=== Haskell Program Error ===")
            print(result.stderr)

    except FileNotFoundError:
        print("Error: Could not find the executable 'PermutationProgram8'.")
        print("Make sure it is compiled and located in the same directory, or provide the full path.")
        sys.exit(1)

if __name__ == "__main__":
    main()
