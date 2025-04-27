import ast
import math
from copy import copy

import astunparse
import openai
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer

assert math

model_name = "text-davinci-002"


def exec_safe(code_str, gvars, lvars):
    banned_phrases = ["import", "__"]
    for phrase in banned_phrases:
        assert phrase not in code_str

    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])
    exec(code_str, custom_gvars, lvars)


default_query_kwargs = {
    "engine": model_name,
    "max_tokens": 512,
    "temperature": 0,
}


def lmp(
    base_prompt,
    query,
    stop_tokens=None,
    log=True,
    return_response=False,
    query_kwargs=None,
):
    new_prompt = f"{base_prompt}\n{query}"
    use_query_kwargs = copy(default_query_kwargs)
    if query_kwargs is not None:
        use_query_kwargs.update(query_kwargs)
    response = openai.Completion.create(
        prompt=new_prompt, stop=stop_tokens, **use_query_kwargs
    )["choices"][0]["text"].strip()

    if log:
        print(query)
        print(response)

    if return_response:
        return response


def lmp_fgen(
    prompt,
    f_name,
    f_sig,
    stop_tokens=["# define function:", "# example:"],
    recurse=False,
    context_vars=None,
    bug_fix=False,
    log=True,
    return_src=False,
    query_kwargs=None,
    info="",
):
    query = f"# define function: {f_sig}."
    if info:
        query = f"{query}\n# info: {info}."
    f_src = lmp(
        prompt,
        query,
        stop_tokens=stop_tokens,
        log=False,
        return_response=True,
        query_kwargs=query_kwargs,
    )
    if bug_fix:
        f_src = openai.Edit.create(
            model="code-davinci-edit-001",
            input="# " + f_src,
            temperature=0,
            instruction="Fix the bug if there is one. Improve readability. Keep same inputs and outputs. Only small changes. No comments.",
        )["choices"][0]["text"].strip()

    if context_vars is None:
        context_vars = {}
    gvars = context_vars
    lvars = {}

    f_success = True
    try:
        exec_safe(f_src, gvars, lvars)
        f = lvars[f_name]
    except Exception as e:
        print(e)
        f = lambda *args, **kargs: None
        f_success = False

    if recurse and f_success:
        # recursively define child_fs in the function body if needed
        f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
        potential_child_fs, potential_child_f_sigs = {}, {}
        f_parser = FunctionParser(potential_child_fs, potential_child_f_sigs)
        f_parser.visit(ast.parse(f_def_body))
        for (
            potential_child_f_name,
            potential_child_f_sig,
        ) in potential_child_f_sigs.items():
            if potential_child_f_name in potential_child_fs:
                potential_child_fs[potential_child_f_name] = potential_child_f_sig

        child_fs, child_f_srcs = {}, {}
        for child_f_name, child_f_sig in potential_child_fs.items():
            all_vars = merge_dicts([context_vars, child_fs])
            if not var_exists(child_f_name, all_vars):
                print(f"creating child function: {child_f_name}")
                child_f, child_f_src = lmp_fgen(
                    prompt,
                    child_f_name,
                    child_f_sig,
                    stop_tokens=stop_tokens,
                    context_vars=all_vars,
                    bug_fix=bug_fix,
                    log=False,
                    recurse=True,
                    return_src=True,
                    query_kwargs=query_kwargs,
                )

                child_fs[child_f_name] = child_f
                child_f_srcs[child_f_name] = child_f_src

        if len(child_fs) > 0:
            # redefine parent f so newly created child_fs are in scope
            gvars = merge_dicts([context_vars, child_fs])
            lvars = {}

            exec_safe(f_src, gvars, lvars)

            f = lvars[f_name]

    if log:
        to_print = highlight(f"{query}\n{f_src}", PythonLexer(), TerminalFormatter())
        print(f"LMP FGEN created:\n\n{to_print}\n")

    if not f_success:
        raise Exception("LMP FGEN failed to create function")

    if return_src:
        return f, f_src
    return f


class FunctionParser(ast.NodeTransformer):
    def __init__(self, fs, f_assigns):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}
