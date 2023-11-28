def insert_debug_info(c_code, cnt):
    lines = c_code.split('\n')
    modified_lines = []
    for line in lines:
        modified_lines.append(line)
        if line.strip().startswith('{') and not line.startswith(' '):
            modified_lines.append('    printk(KERN_ERR "=====================================   %d\\n");' % cnt)
            cnt += 1
    return '\n'.join(modified_lines), cnt

cnt = 529

while True:
    filename = input('Enter filename: ')
    if not filename:
        break

    with open(filename, 'r') as file:
        c_code = file.read()

    modified_code, cnt = insert_debug_info(c_code, cnt)

    with open(filename, 'w') as file:
        file.write(modified_code)
