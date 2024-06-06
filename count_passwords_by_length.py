def count_passwords_by_length(passwords):
    password_length_count = {}
    
    for password in passwords:
        length = len(password)
        if length in password_length_count:
            password_length_count[length] += 1
        else:
            password_length_count[length] = 1
    
    return password_length_count

# Read passwords from a file
file_path = r'D:\smne\final - Copy\myspace\train_myspace.txt'  
passwords = []

with open(file_path, 'r') as file:
    for line in file:
        password = line.rstrip('\n')  # Remove trailing newline character
        passwords.append(password)

# Count passwords by length
password_length_count = count_passwords_by_length(passwords)

# Print the summary in ascending order
for length in sorted(password_length_count.keys()):
    count = password_length_count[length]
    print(f"Number of {length}-character passwords: {count}")
