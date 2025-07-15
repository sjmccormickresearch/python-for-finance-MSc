# QUESTION 1: Prime Numbers

# Function to filter prime numbers from a list
def prime(numbers):
    # checks if the number is prime.
    def is_prime(n):
        if n < 2:
            return False  # If less than 2, it is not a prime number

        for i in range(2, int(n ** 0.5) + 1):  # Check divisibility from 2 to sqrt(n) (rather than n/2) for efficiency
            if n % i == 0:
                return False  # If divisible, it's not prime
        return True  # If no divisors found, it's prime

    return [num for num in numbers if is_prime(num)]  # Filter primes from the list


# TESTING QUESTION 1: Prime Numbers
# Could use f-strings combined with other methods for cleaner displayability, but would require extra code
print("TESTING QUESTION 1: Prime Numbers")
print(f"Input: {[2, 3, 6]} \t\t\t\t\t\t\t Output: {prime([2, 3, 6])}")
print(f"Input: {[4, 6]} \t\t\t\t\t\t\t\t Output: {prime([4, 6])}")
print(f"Input: {[]} \t\t\t\t\t\t\t\t Output: {prime([])}")
print("\n")  # Extra new line for better readability in the output



# QUESTION 2: Abbreviation Function

# Function to extract the first letter of each word in a text
def abbreviation(text: str) -> str:
    letters = ""  # Initialize an empty string to store abbreviation letters

    for i in range(len(text)):  # Iterate through each character in the text
        # take the first character but don't take it if it's a space (allows spaces at beginning of input)
        # take any character if the previous character is a space but the current character is not a space (allows having multiple space in between words)
        # this ensures we pick the first letter of each word while ignoring multiple spaces
        # doing it this way allows us to check only two conditions, and avoid using .split()
        if (i == 0 and text[0] != " ") or (text[i - 1] == " " and text[i] != " "):
            letters += text[i] # Append the selected character to the abbreviation string initialized earlier

    return letters


# 2. TESTING QUESTION 2: Abbreviation Function
print("TESTING QUESTION 2: Abbreviation Function")
print(f"Input: Finance \t\t\t\t\t\t\t\t Output: {abbreviation("Finance")}")
print(f"Input: Journal of Finance \t\t\t\t\t\t Output: {abbreviation("Journal of Finance")}")
print(f"Input: Financial Modelling in Python \t\t\t\t\t Output: {abbreviation("Financial Modelling in Python")}")
print(f"Input:      Financial     Modelling      in      Python \t\t Output: {abbreviation("     Financial     Modelling      in      Python")}")
print("\n")



# QUESTION 3: Returns Function

# Function to calculate percentage returns between daily stock prices
def returns(prices) -> list:
    rates = []  # List to store calculated return percentages

    for i in range(len(prices)):  # iterate through the price list
        if i != len(prices) - 1:  # Ensure we don't calculate return for the last element
            # Calculate percentage return using the formula: ((new_price - old_price) / old_price) * 100
            rate = ((prices[i + 1] - prices[i]) / prices[i]) * 100  # avoids calculating return for the first element, or any element in isolation
            formatted_rate = f"{rate:.2f}%"  # Format the return to 2 decimal places and add % sign
            rates.append(formatted_rate)  # Append the formatted return percentage to the list

    return rates  # Return the list of calculated returns


# TESTING QUESTION 3: Returns Function
print("TESTING QUESTION 3: Returns Function")
# Using .join() to format output as a string separated by commas
print(f"Input: {[100, 120, 150, 170, 100]} \t\t\t\t\t Output: {", ".join(returns([100, 120, 150, 170, 100]))}")
print(f"Input: {[20, 25, 20, 10, 5]} \t\t\t\t\t\t Output: {", ".join(returns([20, 25, 20, 10, 5]))}")
print("\n")



# QUESTION 4: First Missing Positive Function

# Function to sort an array
# Had originally used bubble-sort but this way is more efficient
def first_missing_positive(numbers):
    # Move each number to its correct position in the list
    n = len(numbers)
    for i in range(n):
        while 1 <= numbers[i] <= n and numbers[i] != numbers[numbers[i] - 1]:
            numbers[numbers[i] - 1], numbers[i] = numbers[i], numbers[numbers[i] - 1]

    # After rearranging, check each index
    for i in range(n):
        if numbers[i] != i + 1:
            return i + 1

    # If all numbers from 1 to n are in place, return n + 1
    return n + 1


# TESTING QUESTION 4: First Missing Positive Function
print("TESTING QUESTION 4: First Missing Positive Function")
print(f"Input: {[1, 2, 0]} \t\t\t\t\t\t\t Output: {first_missing_positive([1, 2, 0])}")
print(f"Input: {[3, 4, -1, 1]} \t\t\t\t\t\t\t Output: {first_missing_positive([3, 4, -1, 1])}")
print(f"Input: {[7, 8, 9, 11]} \t\t\t\t\t\t\t Output: {first_missing_positive([7, 8, 9, 11])}")
print(f"Input: {[3, 4, -1, -3, 1]} \t\t\t\t\t\t Output: {first_missing_positive([3, 4, -1, -3, 1])}")
print(f"Input: {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]} \t\t\t\t\t Output: {first_missing_positive([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}")
print("\n")



# QUESTION 5: Average Marks Function

# Function to compute the average mark for each student, allowing for multiple tests
def average_mark(attempts):
    student_data = {}  # Dictionary to store student_id as key and [total_marks, count] as value.

    # Accumulate marks and count steps for each student
    for student_id, mark in attempts:
        if student_id in student_data:
            student_data[student_id][0] += mark  # Add the new mark to the total
            student_data[student_id][1] += 1  # Increment attempt count
        else:
            student_data[student_id] = [mark, 1]  # Initialize entry with first mark and count = 1

    # Step 2: Calculate the average for each student
    result = []
    for student_id, (total_marks, num_attempts) in student_data.items():
        average = total_marks / num_attempts  # Compute average
        result.append([student_id, round(average)])  # Round average and store result

    return result  # Return the list of student averages

# TESTING QUESTION 5: Average Marks Function
print("TESTING QUESTION 5: Average Marks Function")
print(f"Input: {[[3527, 50], [3528, 60], [3527, 54]]} \t\t\t\t Output: {average_mark([[3527, 50], [3528, 60], [3527, 54]])}")