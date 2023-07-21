import sys
import os
# Get the current script's directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory path (project_folder in this case)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to Python's sys.path
sys.path.append(parent_dir)

from ...FollowCar.control import send_to_arduino

if __name__ == "__main__":
    send_to_arduino('a','20')