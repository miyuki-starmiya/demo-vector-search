import math
import random


def generate_random_numbers(num_numbers, upper_limit):
    # Generate the random numbers
    random_numbers = [math.floor(random.uniform(0, upper_limit)) for _ in range(num_numbers)]

    # Format the random numbers as a single string in the desired format
    formatted_numbers = "(" + ", ".join(map(str, random_numbers)) + ")"

    # Write the formatted numbers to output.txt
    with open('output.txt', 'w') as file:
        file.write(formatted_numbers)

    print("Random numbers have been generated and written to output.txt in the desired format")

    return formatted_numbers
