"""Model-generated finite API programs with evidence-fed retries.

This module is independent from the legacy handwritten codegen.py. No Python exec
is used: candidates are interpreted as a deliberately small finite subset (including if, no loops).
The trusted caller owns reset, API implementations, deadlines and evaluation.
A step/call budget cannot interrupt a blocking API; use process isolation for that.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import json
import math
import operator
from pathlib import Path
import re


class PolicyRejected(ValueError):
    pass


class PolicyProgram:
    def __init__(self, source, allowed_names):
        if not isinstance(source, str) or len(source) > 20000:
            raise PolicyRejected("source too long or not text")
        self.source = source
        self.allowed = frozenset(allowed_names)
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise PolicyRejected("expected only def policy(api)")
        fn = tree.body[0]
        args = fn.args
        if (fn.name != "policy" or fn.decorator_list or fn.returns is not None
                or args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs
                or args.defaults or len(args.args) != 1 or args.args[0].arg != "api"
                or args.args[0].annotation is not None):
            raise PolicyRejected("expected unannotated def policy(api)")
        nodes = list(ast.walk(tree))
        if len(nodes) > 1500 or len(fn.body) > 100:
            raise PolicyRejected("AST budget exceeded")
        allowed_types = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign,
                         ast.Expr, ast.Return, ast.Call, ast.Attribute, ast.Name, ast.Load,
                         ast.Store, ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Subscript,
                         ast.Slice, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
                         ast.Div, ast.USub, ast.UAdd, ast.keyword,
                         ast.If, ast.Compare, ast.Is, ast.IsNot, ast.Eq, ast.NotEq,
                         ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or, ast.Not)
        for n in nodes:
            if not isinstance(n, allowed_types):
                raise PolicyRejected(f"unsupported syntax: {type(n).__name__}")
            if isinstance(n, ast.Name) and n.id.startswith("_"):
                raise PolicyRejected("private names forbidden")
            if isinstance(n, ast.Attribute):
                if not (isinstance(n.value, ast.Name) and n.value.id == "api"
                        and n.attr in self.allowed and not n.attr.startswith("_")):
                    raise PolicyRejected(f"unsupported API attribute {ast.unparse(n)}; allowed methods: {sorted(self.allowed)}")
            if isinstance(n, ast.Call):
                if not isinstance(n.func, ast.Attribute):
                    raise PolicyRejected("only api.method calls allowed")
                if any(k.arg is None for k in n.keywords):
                    raise PolicyRejected("keyword expansion forbidden")
            if isinstance(n, ast.Assign):
                if (len(n.targets) != 1 or not isinstance(n.targets[0], ast.Name)
                        or n.targets[0].id == "api"):
                    raise PolicyRejected("assignment must target one local name")
        self.body = fn.body

    def validate_api(self, api_mapping):
        """Check every call signature before any API can execute. No value inference."""
        import inspect
        for statement in self.body:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr
                if name not in api_mapping:
                    raise PolicyRejected(f"unavailable API: {name}")
                signature = inspect.signature(api_mapping[name])
                try:
                    signature.bind(*[None for _ in node.args],
                                   **{k.arg:None for k in node.keywords})
                except TypeError as exc:
                    raise PolicyRejected(f"api.{name}{signature}: received {len(node.args)} positional "
                                         f"arguments and keywords {[k.arg for k in node.keywords]}; {exc}") from exc

    def run(self, api_mapping, call_limit=50):
        self.validate_api(api_mapping)
        values, trace = {}, []
        operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                      ast.Mult: operator.mul, ast.Div: operator.truediv}

        def data_only(value, depth=0):
            # Never let trusted API return a simulation/model object to candidates.
            if depth > 8:
                raise PolicyRejected("API data nesting exceeds limit")
            if value is None or isinstance(value, (str, bool, int, float)):
                if isinstance(value, (int, float)) and (not math.isfinite(value) or abs(value) > 1e9):
                    raise PolicyRejected("non-finite or excessive numeric value")
                if isinstance(value, str) and len(value)>10000:
                    raise PolicyRejected("string exceeds limit")
                return value
            if isinstance(value, (tuple, list)) and len(value) <= 100:
                return tuple(data_only(x,depth+1) for x in value)
            if isinstance(value, dict) and len(value) <= 100 and all(isinstance(k,str) for k in value):
                return {k:data_only(v,depth+1) for k,v in value.items()}
            raise PolicyRejected("API must return bounded plain data, not live objects")

        def evaluate(n):
            if isinstance(n, ast.Constant):
                return data_only(n.value)
            if isinstance(n, ast.Name):
                if n.id not in values:
                    raise PolicyRejected(f"undefined local: {n.id}")
                return values[n.id]
            if isinstance(n, (ast.List, ast.Tuple)):
                return data_only(tuple(evaluate(v) for v in n.elts))
            if isinstance(n, ast.Dict):
                return data_only({evaluate(k):evaluate(v) for k,v in zip(n.keys,n.values)})
            if isinstance(n, ast.Compare):
                value = evaluate(n.left)
                comparisons = {ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,
                    ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge,
                    ast.Is:operator.is_,ast.IsNot:operator.is_not}
                for op,other in zip(n.ops,n.comparators):
                    right = evaluate(other)
                    if not comparisons[type(op)](value,right): return False
                    value=right
                return True
            if isinstance(n, ast.BoolOp):
                value = evaluate(n.values[0])
                for other in n.values[1:]:
                    if (isinstance(n.op,ast.And) and not value) or (isinstance(n.op,ast.Or) and value): return value
                    value=evaluate(other)
                return value
            if isinstance(n, ast.UnaryOp):
                value = evaluate(n.operand)
                if isinstance(n.op,ast.Not): return not value
                if type(value) not in (int,float):
                    raise PolicyRejected("unary arithmetic must be numeric")
                return data_only(-value if isinstance(n.op,ast.USub) else +value)
            if isinstance(n, ast.BinOp):
                a,b = evaluate(n.left), evaluate(n.right)
                if type(a) not in (int,float) or type(b) not in (int,float):
                    raise PolicyRejected("arithmetic must be numeric")
                return data_only(operations[type(n.op)](a,b))
            if isinstance(n, ast.Subscript):
                collection = evaluate(n.value)
                if isinstance(n.slice, ast.Slice):
                    index = slice(*(evaluate(v) if v is not None else None
                                    for v in (n.slice.lower,n.slice.upper,n.slice.step)))
                else:
                    index = evaluate(n.slice)
                return data_only(collection[index])
            if isinstance(n, ast.Call):
                if len(trace)>=call_limit:
                    raise PolicyRejected("API call budget exceeded")
                name = n.func.attr
                if name not in self.allowed or name not in api_mapping:
                    raise PolicyRejected(f"unavailable API: {name}")
                args = [evaluate(x) for x in n.args]
                kwargs = {k.arg:evaluate(k.value) for k in n.keywords}
                if len(trace)>=call_limit:
                    raise PolicyRejected("API call budget exceeded after nested arguments")
                row = {"method":name,"args":deepcopy(args),"kwargs":deepcopy(kwargs),"status":"started"}
                trace.append(row)
                try:
                    result = data_only(api_mapping[name](*args,**kwargs))
                    row.update(status="returned",result=result)
                    return result
                except Exception as exc:
                    row.update(status="failed",error=f"{type(exc).__name__}: {exc}")
                    raise
            raise PolicyRejected(f"unsupported expression: {type(n).__name__}")

        def statements(body):
            for statement in body:
                if isinstance(statement, ast.Assign):
                    values[statement.targets[0].id] = evaluate(statement.value)
                elif isinstance(statement, ast.Expr):
                    evaluate(statement.value)
                elif isinstance(statement, ast.Return):
                    return True, evaluate(statement.value) if statement.value else None
                elif isinstance(statement, ast.If):
                    returned, value = statements(statement.body if evaluate(statement.test) else statement.orelse)
                    if returned: return True,value
                else:
                    raise PolicyRejected("unsupported statement")
            return False,None

        try:
            _, value = statements(self.body)
            return {"trace":trace,"return_value":value}
        except Exception as exc:
            exc.policy_trace = trace
            raise


def extract_source(raw):
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```",raw,re.S)
    source = (fenced.group(1) if fenced else raw).strip()
    tree = ast.parse(source)
    # Mechanical API adapter only: preserve every generated statement; never add
    # task actions, replace coordinates or fix logic. Full AST validation follows.
    if tree.body and not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in tree.body):
        source = "def policy(api):\n" + "\n".join("    " + line for line in source.splitlines())
    return source


class PolicySampler:
    def __init__(self, backend, *, focused_repair=False):
        self.backend = backend
        self.focused_repair = focused_repair

    def synthesize_and_test(self, action_model, api_docs, evaluate, max_attempts=3, out_dir=None,
                            *, initial_source=None, initial_evidence=None):
        """evaluate(program) resets an identical scenario, executes and independently checks.

        The callback must report code_executed/valid_trial/goal_ok/effect_ok explicitly.
        Failed trials never certify an impossible action; h remains immutable.
        This API is not wired into the legacy five-task loop automatically.
        """
        h = deepcopy(action_model)
        attempts=[]
        messages=[{"role":"system","content":
            "Generate ONLY def policy(api): Python source. Use finite straight-line assignments "
            "and calls to the documented api methods. No imports, loops, other functions, "
            "object attributes, simulator access, or success flags. Return value does not determine success. "
            "Do not change the action model. Use previous execution evidence to repair CODE only."},
            {"role":"user","content":json.dumps({"action_model":h,"api_contracts":api_docs},ensure_ascii=False)}]
        if self.focused_repair:
            messages = [{"role":"system", "content":
                "You are a Python robotics programmer. Write def policy(api): with local variables "
                "and the exact API signatures below. No imports or loops. A coordinate is ONE tuple "
                "argument, not multiple scalar arguments. Assign function return values before using "
                "them. Do not fabricate API methods. Do not change the STRIPS contract."},
                {"role":"user", "content":"Implement this ONE action:\n" +
                 (h.get("description") or json.dumps(h,ensure_ascii=False))
                 + "\nAPI reference (all coordinates are Python tuples, never dictionaries):\n" + "\n".join(api_docs.values())
                 + "\nOutput a single Python function. Use named local variables for pose and return values. "
                 "Only use documented argument names; positional arguments are also allowed. "
                 "The payload being held must not be used as its own placement destination."}]
        initial_messages = deepcopy(messages)
        if initial_source is not None:
            messages.extend([{"role":"assistant","content":initial_source},
                {"role":"user","content":"Repair this actual failed program using the API reference. "
                 "Output a corrected full def policy(api) only. Observed failure:\n"+
                 json.dumps(initial_evidence or {},ensure_ascii=False)}])
        for index in range(max_attempts):
            row={"attempt":index+1,"backend":getattr(self.backend,"name",type(self.backend).__name__),
                 "source_type":"backend_generated", "messages":deepcopy(messages)}
            try:
                temperature = (min(0.2*index, 0.6) if self.focused_repair else 0.0)
                row["temperature"] = temperature
                raw=self.backend.chat(messages,max_tokens=1200,temperature=temperature)
                row["raw"] = raw
                source=extract_source(raw)
                row.update(raw=raw,source=source,sha256=hashlib.sha256(source.encode()).hexdigest(),
                           normalization="remove optional code fence; wrap top-level statements in policy(api); no action edits")
                program=PolicyProgram(source,api_docs.keys())
                verdict=evaluate(program)
                if not isinstance(verdict,dict):
                    raise ValueError("evaluator must return structured evidence")
                row["evaluation"]=deepcopy(verdict)
                row["accepted"]=all(verdict.get(k) is True for k in
                                    ("code_executed","valid_trial","goal_ok","effect_ok"))
            except Exception as exc:
                row.update(accepted=False,evaluation={"diagnosis_status":"unresolved",
                    "exception_type":type(exc).__name__,"exception":str(exc),
                    "trace":getattr(exc,"policy_trace",[]),"candidate_model_patch":None})
            attempts.append(row)
            if row["accepted"]:
                break
            if self.focused_repair:
                ev = row["evaluation"]
                facts = ev.get("feedback",{}).get("observed_facts",{})
                compact = {"exception":ev.get("exception") or facts.get("exception"),
                           "unmet_effects":facts.get("gt_violations",[]),
                           "last_calls":ev.get("trace",[])[-12:],
                           "diagnostics":ev.get("diagnostics",{}),
                           "valid_trial":ev.get("valid_trial"),"goal_ok":ev.get("goal_ok"),
                           "effect_ok":ev.get("effect_ok")}
                # Keep the immutable task/API contract and ONLY the last failed program.
                # No replacement code or target action sequence is supplied by the host.
                messages = deepcopy(initial_messages) + [
                    {"role":"assistant","content":row.get("raw", "No candidate was returned.")},
                    {"role":"user","content":"The above program was rejected. Fix the exact error; "
                     "do not repeat the same invalid calls. Preserve the action contract. "
                     "Return the full corrected function only. Observed evidence:\n"+
                     json.dumps(compact,ensure_ascii=False)}]
            else:
                messages.extend([{"role":"assistant","content":row.get("raw", "No candidate was returned.")},
                    {"role":"user","content":"Repair the previous code using this observed evidence. "
                     "Other causes remain unresolved. Keep the same action model and APIs.\n"+
                     json.dumps(row["evaluation"],ensure_ascii=False)}])
        result={"success":bool(attempts and attempts[-1]["accepted"]),"action_model":h,
                "attempts":attempts,"action_model_modified":False,"integration":"standalone_sampler",
                "initial_source":initial_source,"initial_evidence":initial_evidence,
                "note":"Backend identity must be reported; fixture backends are not real model evidence."}
        if out_dir is not None:
            path=Path(out_dir); path.mkdir(parents=True,exist_ok=True)
            (path/"policy_sampling.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        return result
