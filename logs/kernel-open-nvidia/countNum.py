filename = input("Enter your log file name: ")

# 用于存储出现过的数字
numbers = set()

# 读取日志文件
with open(filename, "r") as file:
    for line in file:
        # 分割每行，并取最后一个元素作为数字
        parts = line.strip().split()
        if parts:
            number = parts[-1]
            # 确保是数字
            if number.isdigit():
                numbers.add(int(number))

# 输出结果
print("Unique numbers found in the log:")
for num in sorted(numbers):
    print(num)
