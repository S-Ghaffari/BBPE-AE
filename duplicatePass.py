password_counts = {}
total=0
with open("Generated Passwords-10epoch-hotmail-1000.txt", 'r', encoding='UTF-8',errors='ignore') as file:
    for line in file:
        password = line.strip()
        if password in password_counts:
            password_counts[password] += 1
        else:
            password_counts[password] = 1
            
duplicate_total=sum(1 for count in  password_counts.values() if count>1)          
for password, count in password_counts.items():
    if count > 1:
        print(f"Password: {password}, Count: {count}")                              
print("total: ",duplicate_total)
