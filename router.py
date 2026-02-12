"""
router.py - neuro-symbolic logic router
routes math/code to python eval, everything else to LLM.
"""

import re
import math
from typing import Tuple, Optional, Callable, Any
from enum import Enum


class QueryType(Enum):
    MATH = "math"
    CODE = "code"
    FACTUAL = "factual"
    CREATIVE = "creative"
    UNKNOWN = "unknown"


class LogicRouter:
    """routes queries: math->eval, code->exec, facts->db, rest->LLM"""

    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*\/\^\%]\s*\d+',
        r'(calculate|compute|what is)\s+\d+',
        r'sqrt|sin|cos|tan|log|exp|pow',
        r'\d+\s*(squared|cubed)',
        r'factorial|fibonacci|prime',
        r'(sum|average|mean|median)\s+of',
    ]
    CODE_PATTERNS = [
        r'(run|execute|eval)\s+(this|the)?\s*(code|python)',
        r'`{3}python',
        r'def\s+\w+\s*\(',
        r'import\s+\w+',
        r'print\s*\(',
    ]
    FACTUAL_PATTERNS = [
        r'(what|when|where|who)\s+(is|was|are|were)\s+',
        r'define\s+\w+',
        r'(capital|population|founder)\s+of',
    ]

    def __init__(self, llm_fallback=None, enable_code_exec=False):
        self.llm_fallback = llm_fallback
        self.enable_code_exec = enable_code_exec

        self.math_regex = [re.compile(p, re.IGNORECASE) for p in self.MATH_PATTERNS]
        self.code_regex = [re.compile(p, re.IGNORECASE) for p in self.CODE_PATTERNS]
        self.factual_regex = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]

        self.facts = {
            "pi": "3.14159265359",
            "e": "2.71828182846",
            "speed of light": "299,792,458 m/s",
            "gravity": "9.81 m/s^2",
        }

        code_status = "ENABLED" if enable_code_exec else "disabled"
        print(f"-- router loaded ({len(self.MATH_PATTERNS)} math patterns, code exec:{code_status}) --")

    def classify(self, query):
        """detect query type from patterns"""
        ql = query.lower()
        for r in self.math_regex:
            if r.search(ql): return QueryType.MATH
        for r in self.code_regex:
            if r.search(ql): return QueryType.CODE
        for r in self.factual_regex:
            if r.search(ql): return QueryType.FACTUAL
        return QueryType.CREATIVE

    def route(self, query):
        """route query -> (result, handled)"""
        qt = self.classify(query)

        if qt == QueryType.MATH:
            return self._handle_math(query), True
        elif qt == QueryType.CODE and self.enable_code_exec:
            return self._handle_code(query), True
        elif qt == QueryType.FACTUAL:
            r = self._handle_factual(query)
            if r: return r, True

        if self.llm_fallback:
            return self.llm_fallback(query), True
        return None, False

    def _handle_math(self, query):
        """eval math with safe namespace"""
        try:
            expr = query.lower()
            expr = re.sub(r'\?', '', expr)
            expr = re.sub(r'what is|calculate|compute|equals?', '', expr)
            expr = re.sub(r'squared', '** 2', expr)
            expr = re.sub(r'cubed', '** 3', expr)
            expr = re.sub(r'to the power of', '**', expr)
            expr = re.sub(r'times', '*', expr)
            expr = re.sub(r'divided by', '/', expr)
            expr = re.sub(r'plus', '+', expr)
            expr = re.sub(r'minus', '-', expr)
            expr = re.sub(r'[^\d\+\-\*\/\.\^\(\)\s]', '', expr)
            expr = expr.replace('^', '**').strip()

            if not expr:
                return "couldn't parse math expression"

            allowed = {
                'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
                'tan': math.tan, 'log': math.log, 'exp': math.exp,
                'pow': pow, 'abs': abs, 'round': round,
                'pi': math.pi, 'e': math.e,
            }
            result = eval(expr, {"__builtins__": {}}, allowed)

            if isinstance(result, float):
                result = int(result) if result == int(result) else round(result, 6)
            return f"{expr} = {result}"
        except Exception as e:
            return f"math error: {e}"

    def _handle_code(self, query):
        """execute python code block (dangerous)"""
        if not self.enable_code_exec:
            return "code execution disabled"
        try:
            m = re.search(r'```python\s*(.*?)\s*```', query, re.DOTALL)
            if not m:
                return "provide code in ```python``` block"
            ns = {}
            exec(m.group(1), {"__builtins__": __builtins__}, ns)
            if 'result' in ns:
                return f"result: {ns['result']}"
            return "code executed"
        except Exception as e:
            return f"code error: {e}"

    def _handle_factual(self, query):
        """lookup from local facts db"""
        ql = query.lower()
        for key, val in self.facts.items():
            if key in ql:
                return f"{key}: {val}"
        return None


# ---- test ----
if __name__ == "__main__":
    print("=" * 60)
    print("router test")
    print("=" * 60)

    router = LogicRouter()

    tests = [
        "What is 2 + 2?",
        "Calculate 15 * 7",
        "What is 10 squared?",
        "What is the speed of light?",
        "Tell me a joke",
    ]

    for q in tests:
        qt = router.classify(q)
        result, handled = router.route(q)
        print(f"\n   Q: {q}")
        print(f"   type: {qt.value}, handled: {handled}")
        if result:
            print(f"   -> {result}")

    print("\ndone")
