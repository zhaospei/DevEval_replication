import json
import os
import pdb
import textwrap
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from concurrent.futures import as_completed, ProcessPoolExecutor

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str) 
    parser.add_argument('--completion_file', type=str) 
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--data_file', type=str, default='/drive2/tuandung/DevEval/data_clean.jsonl')
    return parser

args = get_parser().parse_args()
assert os.path.exists(args.completion_file), f"Path {args.completion_file} does not exist."
wrong_code = "    pass\n"

parser = Parser()
# PY_LANGUAGE = Language('build/my-languages.so', 'python')
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)
# parser.set_language(PY_LANGUAGE)

def count_indent(code):
    if type(code) == str: # a single statement
        return len(code) - len(textwrap.dedent(code))
    elif type(code) == list: # a list of statements, i.e., a function body
        for line in code:
            if line.strip() != '':
                return len(line) - len(textwrap.dedent(line))
            

def adjust_indent(code, new_indent):
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code

def extract_code_from_response(completion: str):
    """
    Extract code from a completion. The code is in markdown format.
    :param completion: String.
    :return: Code in the completion.
    """
    if args.model_type == 'glm':
        completion_lines = completion.split("\\n")
        code_lines = completion.split("\\n")
    else:
        completion_lines = completion.split("\n")
        code_lines = completion.split("\n")
    code_sol, code_eol = None, None
    for i, line in enumerate(completion_lines):
        if line.startswith("```"):
            if code_sol is None:
                code_sol = i+1
            else:
                code_eol = i
                break
    if code_sol is None: # No markdown code block
        if code_eol is None:
            code_sol = 0
            code_eol = len(completion_lines)
        else:
            code_sol = 0
    elif code_eol is None: # No end of markdown block
        code_eol = len(completion_lines)
    code_lines = completion_lines[code_sol:code_eol]
    code = "\n".join(code_lines)
    if args.model_type == 'glm':
        code = code.replace('\\\"', '"')
    return code


def find_function_body(code, function_name):
    # if type(code) == str:
    #     code = bytes(code, "utf8")
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    def search_function(node):
        # 查找类型为function_definition且名称匹配的节点
        for child in node.children:
            if child.type == 'function_definition':
                # 获取函数名
                name_node = child.child_by_field_name('name')
                if name_node and name_node.text.decode('utf8') == function_name:
                    # 找到函数，返回其body部分
                    body_node = child.child_by_field_name('body')
                    return body_node
            # 递归搜索所有子节点
            result = search_function(child)
            if result:
                return result
        return None

    def search_import(node):
        """
        Search all import or from_import nodes from a root node.
        """
        import_nodes = []
        for child in node.children:
            if child.type == 'import_statement' or child.type == 'import_from_statement':
                import_nodes.append(child)
            result = search_import(child)
            if result:
                import_nodes.extend(result)
        return import_nodes

    function_body = search_function(root_node)
    import_nodes = search_import(root_node)
    if function_body:
        if len(function_body.children) == 0: # empty body
            return None
        first_node = function_body.children[0]
        if first_node.type == 'expression_statement' and first_node.children[0].type == 'string':
            start_idx = first_node.end_point[0] + 1
        else:
            start_idx = function_body.start_point[0]
        end_idx = function_body.end_point[0] + 1

        code_lines = code.split("\n")
        body_code = code_lines[start_idx:end_idx]
        if len(body_code) == 0: # empty body
            return None
        if len(import_nodes) > 0:
            body_indent = count_indent(body_code)
            for node in import_nodes:
                _import = ' '*body_indent + '\n'.join(code_lines[node.start_point[0]:(node.end_point[0] + 1)])
                body_code = [_import] + body_code
                # print(_import)
        body_code = "\n".join(body_code)
        return body_code
    else: # no function found
        return None


def extract_body_gpt(data: dict):
    """
    Parse code using the `ast` library and extract its body.
    :param code: String. Code in markdown format.
    :return: Body in the code.
    """
    if len(data['task_id']) == 0:
        raise ValueError(f"Completion is empty: {data['task_id']}")
    
    results = []
    function_name = data['task_id'].split('.')[-1]
    if type(data['generation']) != str:
        data['generation'] = data['generation'][0]
    completion = data['generation']
    code = extract_code_from_response(completion)
    
    if not f'def {function_name}(' in code:
        completion_lines = code.strip('\n').split("\n")
        body_indent = count_indent(completion_lines[0])
        new_complation = []
        for line in completion_lines:
            if count_indent(line) >= body_indent or line.strip() == "":
                new_complation.append(line)
            else:
                break
        new_complation = "\n".join(new_complation)
        results.append(new_complation)
    else:
        function_body = find_function_body(code, function_name)
        if function_body is None:
            results.append(wrong_code)
        else:
            results.append(function_body)
        
    return data['task_id'], results


def extract_body_lm(data: dict):
    results = []
    if type(data['completions']) == str:
        data['completions'] = [data['completions']]
    for completion in data['completions']:
        completion_lines = completion.strip("\u0407").strip('\n').split("\n")
        body_first_line = completion_lines[0]
        body_indent = count_indent(body_first_line)
        new_complation = []
        for line in completion_lines:
            if count_indent(line) >= body_indent or line.strip() == "":
                new_complation.append(line)
            else:
                break
        new_complation = "\n".join(new_complation)
        results.append(new_complation)

    return data['namespace'], results


if __name__ == "__main__":

    extraction = {'gpt': extract_body_gpt, 'lm': extract_body_lm}
    extract_function = extraction[args.model_type]

    benchmark = dict()
    with open(args.data_file, "r") as f:
        for line in f:
            data = json.loads(line)
            benchmark[data['namespace']] = data

    idx = 0
    completion_lines = pd.read_parquet(args.completion_file).to_dict(orient='records') 
    cleaned_code_list = []
    # with open(args.completion_file, "r") as f_r, open(args.output_file, "w") as f_w:
    for data in tqdm(completion_lines):
        # data = json.loads(line)
        namespace, completions = extract_function(data)
        cleaned_code_list.append(completions[0])
        # if namespace in benchmark:
        #     # if type(completions) == str:
        #     #     completions = [completions]
        #     # for completion in completions:
        #     completion = completions[0]
        #     # benchmark[namespace]['completion'] = completion
        #     # benchmark[namespace]['idx'] = idx
        #     f_w.write(json.dumps(benchmark[namespace]) + "\n")
        #     idx += 1
        #     del benchmark[namespace]
            # break
    output_file = args.completion_file.replace('.parquet', '_clean_code.parquet')
    results = pd.DataFrame(completion_lines)
    results['cleaned_code'] = cleaned_code_list
    results.to_parquet(output_file)
    