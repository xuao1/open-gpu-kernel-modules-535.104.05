filename = input("Enter your log file name: ")

number_counts = {}

# 读取日志文件
with open(filename, "r") as file:
    for line in file:
        parts = line.strip().split()
        if parts:
            number = parts[-1]
            if number.isdigit():
                number = int(number)
                number_counts[number] = number_counts.get(number, 0) + 1

# 输出结果
print("Numbers found in the log and their counts:")
for num, count in sorted(number_counts.items()):
    print(f"Number: {num}, Count: {count}")
