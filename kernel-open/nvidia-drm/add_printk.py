import os

def insert_debug_info(c_code, cnt):
    lines = c_code.split('\n')
    modified_lines = []
    in_function = False
    potential_function_start = False

    for i, line in enumerate(lines):
        modified_lines.append(line)

        if ')' in line:
            potential_function_start = True
        elif potential_function_start and line.strip().startswith('{'):
            modified_lines.append('    printk(KERN_ERR "nvidia-drm =====================================   %d\\n", ' + str(cnt) + ');')
            cnt += 1
            potential_function_start = False
        elif not line.strip().startswith('{'):
            potential_function_start = False

    return '\n'.join(modified_lines), cnt

def process_directory(directory, cnt):
    for filename in os.listdir(directory):
        if filename.endswith('.c'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                c_code = file.read()

            modified_code, cnt = insert_debug_info(c_code, cnt)

            with open(file_path, 'w') as file:
                file.write(modified_code)
    return cnt

cnt = 1
cnt = process_directory('.', cnt)