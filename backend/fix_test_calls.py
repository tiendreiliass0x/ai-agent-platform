#!/usr/bin/env python3
"""
Script to update all create_agent() calls in test_service_layer.py
to use the new signature with db session parameter.
"""

import re

def fix_test_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Pattern to match create_agent calls
    # Matches: agent = await service.create_agent(\n            user_id=...
    # Or: agent = await db_service.create_agent(\n            user_id=...
    pattern = r'([ ]+)(agent|original_agent|valid_agent) = await (service|db_service)\.create_agent\(\s*\n(\s+)user_id='

    def replacement(match):
        indent = match.group(1)
        var_name = match.group(2)
        service_name = match.group(3)
        param_indent = match.group(4)

        return f'''{indent}async with await get_db_session() as db:
{indent}    {var_name} = await {service_name}.create_agent(
{param_indent}db=db,
{param_indent}user_id='''

    content = re.sub(pattern, replacement, content)

    # Now we need to add the db.commit() and proper closing
    # Find all the patterns we just created and add commit after the closing parenthesis
    lines = content.split('\n')
    new_lines = []
    in_create_agent_block = False
    block_indent = ''

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line starts a create_agent block we modified
        if 'async with await get_db_session() as db:' in line and i + 1 < len(lines) and 'create_agent(' in lines[i + 1]:
            new_lines.append(line)
            in_create_agent_block = True
            block_indent = line[:line.index('async')]
            i += 1
            continue

        if in_create_agent_block:
            new_lines.append(line)
            # Look for the closing parenthesis of create_agent
            if ')' in line and 'create_agent' not in line:
                # Add commit line
                new_lines.append(f'{block_indent}    await db.commit()')
                in_create_agent_block = False
        else:
            new_lines.append(line)

        i += 1

    return '\n'.join(new_lines)

if __name__ == '__main__':
    file_path = 'tests/test_service_layer.py'
    fixed_content = fix_test_file(file_path)

    with open(file_path, 'w') as f:
        f.write(fixed_content)

    print(f"Fixed {file_path}")
