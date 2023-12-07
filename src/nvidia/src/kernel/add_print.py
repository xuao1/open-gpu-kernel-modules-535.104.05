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
            modified_lines.append('    NV_PRINTF(LEVEL_ERROR, "############### src/nvidia/src/kernel %d\\n", ' + str(cnt) + ');')
            cnt += 1
            potential_function_start = False
        elif not line.strip().startswith('{'):
            potential_function_start = False

    return '\n'.join(modified_lines), cnt

def process_directory(directory, cnt):
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            cnt = process_directory(full_path, cnt)
        elif item.endswith('.c'):
            with open(full_path, 'r') as file:
                c_code = file.read()

            modified_code, cnt = insert_debug_info(c_code, cnt)

            with open(full_path, 'w') as file:
                file.write(modified_code)
    return cnt

cnt = 1
cnt = process_directory('.', cnt)
