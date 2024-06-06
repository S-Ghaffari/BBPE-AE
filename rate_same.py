count = 0
with open("test_myspace.txt", 'r', encoding='UTF-8', errors='ignore') as file1:
    with open("Generated Passwords2000.txt", 'rt', encoding='UTF-8', errors='ignore') as file2:
        data1 = file1.read().split()
        data2 = file2.read().split()
        print('Number of words in text file(dataset) :', len(data1))
        print('Number of words in text file(generated_pass) :', len(data2))


        same = set(file1).intersection(file2)               
        
with open('same_sample_my2000.txt', 'w') as file_out:
    for line in same:
        count += 1  # Increment the counter
        print(line)
        #file_out.write(line)

print("Number of identical passwords obtained:", count)


print("-"*60)
