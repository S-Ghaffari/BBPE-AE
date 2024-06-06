def count_passwords_by_length(passwords):
    password_length_count = {}
        for password in passwords:
        length = len(password)
        if length in password_length_count:
            password_length_count[length] += 1
        else:
            password_length_count[length] = 1
        return password_length_count

file_path = 'train_myspace.txt'  
passwords = []

with open(file_path, 'r') as file:
    for line in file:
        password = line.rstrip('\n') 
        passwords.append(password)

password_length_count = count_passwords_by_length(passwords)
for length in sorted(password_length_count.keys()):
    count = password_length_count[length]
    print(f"Number of {length}-character passwords: {count}")
