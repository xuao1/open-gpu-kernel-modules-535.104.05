import re

filename = input('Enter your log file name: ')

# 读取日志文件
with open(filename, 'r') as file:
    lines = file.readlines()

# 定义正则表达式来匹配时间戳
timestamp_pattern = r'\[\s*\d+\.\d{6}\]'

# 使用正则表达式替换时间戳为空字符串
lines_without_timestamp = [re.sub(timestamp_pattern, '', line) for line in lines]

# 将处理后的日志写回文件
with open(filename, 'w') as file:
    file.writelines(lines_without_timestamp)
