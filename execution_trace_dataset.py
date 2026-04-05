from __future__ import annotations

import ast
import json
import random
import string
import sys
from dataclasses import asdict, dataclass, field
from typing import Literal


VariableName = str
BinaryOpName = Literal["+", "-", "*"]
CompareOpName = Literal["<", "<=", ">", ">=", "==", "!="]
TaskFamily = Literal["straight", "branch", "loop", "nested", "mixed"]
ConcreteTaskFamily = Literal["straight", "branch", "loop", "nested"]


@dataclass(frozen=True)
class ConstExpr:
    value: int


@dataclass(frozen=True)
class VarExpr:
    name: VariableName


@dataclass(frozen=True)
class BinOpExpr:
    op: BinaryOpName
    left: "Expr"
    right: "Expr"


Expr = ConstExpr | VarExpr | BinOpExpr


@dataclass(frozen=True)
class CompareExpr:
    op: CompareOpName
    left: Expr
    right: Expr


@dataclass(frozen=True)
class AssignStmt:
    target: VariableName
    expr: Expr


@dataclass(frozen=True)
class IfStmt:
    cond: CompareExpr
    then_body: tuple["Stmt", ...]
    else_body: tuple["Stmt", ...]


@dataclass(frozen=True)
class WhileStmt:
    cond: CompareExpr
    body: tuple["Stmt", ...]


@dataclass(frozen=True)
class EmitStmt:
    expr: Expr


Stmt = AssignStmt | IfStmt | WhileStmt | EmitStmt


@dataclass(frozen=True)
class Program:
    body: tuple[Stmt, ...]


@dataclass(frozen=True)
class Instruction:
    op: str
    arg: int | str | None = None
    arg2: int | str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"op": self.op}
        if self.arg is not None:
            payload["arg"] = self.arg
        if self.arg2 is not None:
            payload["arg2"] = self.arg2
        return payload


@dataclass
class TraceEvent:
    step: int
    pc: int
    opcode: str
    arg: int | str | None
    arg2: int | str | None
    step_type: str
    stack_before: list[int]
    stack_after: list[int]
    env_before: dict[str, int]
    env_after: dict[str, int]
    env_delta: dict[str, int]
    memory_reads: list[str]
    memory_writes: dict[str, int]
    output_before: list[int]
    output_after: list[int]
    output_delta: list[int]
    branch_taken: bool | None
    text: str


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 0
    max_statements: int = 6
    max_depth: int = 2
    max_trace_steps: int = 128
    task_family: TaskFamily = "mixed"
    task_family_mixture: tuple[tuple[str, float], ...] = ()
    const_min: int = -4
    const_max: int = 9
    loop_counter_min: int = 2
    loop_counter_max: int = 5
    variables: tuple[str, ...] = tuple(string.ascii_lowercase[:8])


@dataclass
class ExampleBuilder:
    rng: random.Random
    config: GenerationConfig
    next_id: int = 0

    def fresh_id(self) -> str:
        self.next_id += 1
        return f"exec-{self.next_id:07d}"


def _expr_to_source(expr: Expr) -> str:
    if isinstance(expr, ConstExpr):
        return str(expr.value)
    if isinstance(expr, VarExpr):
        return expr.name
    return f"({_expr_to_source(expr.left)} {expr.op} {_expr_to_source(expr.right)})"


def _cond_to_source(cond: CompareExpr) -> str:
    return f"{_expr_to_source(cond.left)} {cond.op} {_expr_to_source(cond.right)}"


def _stmt_to_source(stmt: Stmt, indent: int = 0) -> list[str]:
    prefix = "    " * indent
    if isinstance(stmt, AssignStmt):
        return [f"{prefix}{stmt.target} = {_expr_to_source(stmt.expr)}"]
    if isinstance(stmt, EmitStmt):
        return [f"{prefix}emit {_expr_to_source(stmt.expr)}"]
    if isinstance(stmt, IfStmt):
        lines = [f"{prefix}if {_cond_to_source(stmt.cond)}:"]
        for child in stmt.then_body:
            lines.extend(_stmt_to_source(child, indent + 1))
        lines.append(f"{prefix}else:")
        for child in stmt.else_body:
            lines.extend(_stmt_to_source(child, indent + 1))
        return lines
    if isinstance(stmt, WhileStmt):
        lines = [f"{prefix}while {_cond_to_source(stmt.cond)}:"]
        for child in stmt.body:
            lines.extend(_stmt_to_source(child, indent + 1))
        return lines
    raise TypeError(f"Unsupported stmt {type(stmt)!r}")


def _stmt_to_python(stmt: Stmt, indent: int = 0) -> list[str]:
    prefix = "    " * indent
    if isinstance(stmt, AssignStmt):
        return [f"{prefix}{stmt.target} = {_expr_to_source(stmt.expr)}"]
    if isinstance(stmt, EmitStmt):
        return [f"{prefix}print({_expr_to_source(stmt.expr)})"]
    if isinstance(stmt, IfStmt):
        lines = [f"{prefix}if {_cond_to_source(stmt.cond)}:"]
        for child in stmt.then_body:
            lines.extend(_stmt_to_python(child, indent + 1))
        lines.append(f"{prefix}else:")
        for child in stmt.else_body:
            lines.extend(_stmt_to_python(child, indent + 1))
        return lines
    if isinstance(stmt, WhileStmt):
        lines = [f"{prefix}while {_cond_to_source(stmt.cond)}:"]
        for child in stmt.body:
            lines.extend(_stmt_to_python(child, indent + 1))
        return lines
    raise TypeError(f"Unsupported stmt {type(stmt)!r}")


def program_to_source(program: Program) -> str:
    lines: list[str] = []
    for stmt in program.body:
        lines.extend(_stmt_to_source(stmt))
    return "\n".join(lines)


def program_to_python_source(program: Program) -> str:
    lines: list[str] = []
    for stmt in program.body:
        lines.extend(_stmt_to_python(stmt))
    return "\n".join(lines)


def _expr_to_ast(expr: Expr) -> dict[str, object]:
    if isinstance(expr, ConstExpr):
        return {"type": "const", "value": expr.value}
    if isinstance(expr, VarExpr):
        return {"type": "var", "name": expr.name}
    return {
        "type": "binop",
        "op": expr.op,
        "left": _expr_to_ast(expr.left),
        "right": _expr_to_ast(expr.right),
    }


def _cond_to_ast(cond: CompareExpr) -> dict[str, object]:
    return {
        "type": "compare",
        "op": cond.op,
        "left": _expr_to_ast(cond.left),
        "right": _expr_to_ast(cond.right),
    }


def _stmt_to_ast(stmt: Stmt) -> dict[str, object]:
    if isinstance(stmt, AssignStmt):
        return {"type": "assign", "target": stmt.target, "expr": _expr_to_ast(stmt.expr)}
    if isinstance(stmt, EmitStmt):
        return {"type": "emit", "expr": _expr_to_ast(stmt.expr)}
    if isinstance(stmt, IfStmt):
        return {
            "type": "if",
            "cond": _cond_to_ast(stmt.cond),
            "then": [_stmt_to_ast(child) for child in stmt.then_body],
            "else": [_stmt_to_ast(child) for child in stmt.else_body],
        }
    if isinstance(stmt, WhileStmt):
        return {
            "type": "while",
            "cond": _cond_to_ast(stmt.cond),
            "body": [_stmt_to_ast(child) for child in stmt.body],
        }
    raise TypeError(f"Unsupported stmt {type(stmt)!r}")


def program_to_ast(program: Program) -> dict[str, object]:
    return {"type": "program", "body": [_stmt_to_ast(stmt) for stmt in program.body]}


def _push_expr(expr: Expr, instructions: list[Instruction]) -> None:
    if isinstance(expr, ConstExpr):
        instructions.append(Instruction("PUSH_CONST", expr.value))
        return
    if isinstance(expr, VarExpr):
        instructions.append(Instruction("LOAD", expr.name))
        return
    _push_expr(expr.left, instructions)
    _push_expr(expr.right, instructions)
    instructions.append(
        Instruction({"+": "ADD", "-": "SUB", "*": "MUL"}[expr.op])
    )


def _push_compare(cond: CompareExpr, instructions: list[Instruction]) -> None:
    _push_expr(cond.left, instructions)
    _push_expr(cond.right, instructions)
    instructions.append(
        Instruction(
            {
                "<": "CMP_LT",
                "<=": "CMP_LE",
                ">": "CMP_GT",
                ">=": "CMP_GE",
                "==": "CMP_EQ",
                "!=": "CMP_NE",
            }[cond.op]
        )
    )


def _compile_stmt(stmt: Stmt, instructions: list[Instruction]) -> None:
    if isinstance(stmt, AssignStmt):
        _push_expr(stmt.expr, instructions)
        instructions.append(Instruction("STORE", stmt.target))
        return
    if isinstance(stmt, EmitStmt):
        _push_expr(stmt.expr, instructions)
        instructions.append(Instruction("OUTPUT"))
        return
    if isinstance(stmt, IfStmt):
        _push_compare(stmt.cond, instructions)
        jump_false_idx = len(instructions)
        instructions.append(Instruction("JMP_IF_FALSE", None))
        for child in stmt.then_body:
            _compile_stmt(child, instructions)
        jump_end_idx = len(instructions)
        instructions.append(Instruction("JMP", None))
        else_target = len(instructions)
        instructions[jump_false_idx] = Instruction("JMP_IF_FALSE", else_target)
        for child in stmt.else_body:
            _compile_stmt(child, instructions)
        end_target = len(instructions)
        instructions[jump_end_idx] = Instruction("JMP", end_target)
        return
    if isinstance(stmt, WhileStmt):
        loop_start = len(instructions)
        _push_compare(stmt.cond, instructions)
        jump_false_idx = len(instructions)
        instructions.append(Instruction("JMP_IF_FALSE", None))
        for child in stmt.body:
            _compile_stmt(child, instructions)
        instructions.append(Instruction("JMP", loop_start))
        loop_end = len(instructions)
        instructions[jump_false_idx] = Instruction("JMP_IF_FALSE", loop_end)
        return
    raise TypeError(f"Unsupported stmt {type(stmt)!r}")


def compile_program(program: Program) -> list[Instruction]:
    instructions: list[Instruction] = []
    for stmt in program.body:
        _compile_stmt(stmt, instructions)
    instructions.append(Instruction("HALT"))
    return instructions


def _eval_binary(op: str, left: int, right: int) -> int:
    if op == "ADD":
        return left + right
    if op == "SUB":
        return left - right
    if op == "MUL":
        return left * right
    raise ValueError(f"Unsupported binary op {op!r}")


def _eval_compare(op: str, left: int, right: int) -> int:
    if op == "CMP_LT":
        return int(left < right)
    if op == "CMP_LE":
        return int(left <= right)
    if op == "CMP_GT":
        return int(left > right)
    if op == "CMP_GE":
        return int(left >= right)
    if op == "CMP_EQ":
        return int(left == right)
    if op == "CMP_NE":
        return int(left != right)
    raise ValueError(f"Unsupported compare op {op!r}")


def _sorted_env(env: dict[str, int]) -> dict[str, int]:
    return {key: int(env[key]) for key in sorted(env)}


def _env_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for key, value in after.items():
        if before.get(key) != value:
            delta[key] = value
    for key in before:
        if key not in after:
            delta[key] = 0
    return delta


def _event_text(
    opcode: str,
    arg: int | str | None,
    arg2: int | str | None,
    stack_before: list[int],
    stack_after: list[int],
    env_delta: dict[str, int],
    branch_taken: bool | None,
    output_delta: list[int],
) -> str:
    suffix: list[str] = []
    if arg is not None:
        suffix.append(f"arg={arg}")
    if arg2 is not None:
        suffix.append(f"arg2={arg2}")
    if env_delta:
        suffix.append(f"env_delta={env_delta}")
    if output_delta:
        suffix.append(f"output_delta={output_delta}")
    if branch_taken is not None:
        suffix.append(f"branch_taken={int(branch_taken)}")
    suffix.append(f"stack={stack_before}->{stack_after}")
    return f"{opcode} " + " ".join(str(part) for part in suffix)


def execute_program(
    instructions: list[Instruction],
    *,
    max_steps: int = 128,
) -> tuple[list[TraceEvent], dict[str, int], list[int], str]:
    env: dict[str, int] = {}
    stack: list[int] = []
    output: list[int] = []
    pc = 0
    step = 0
    trace: list[TraceEvent] = []
    halt_reason = "halt"

    while 0 <= pc < len(instructions):
        if step >= max_steps:
            halt_reason = "step_limit"
            break
        instr = instructions[pc]
        env_before = _sorted_env(env)
        stack_before = list(stack)
        output_before = list(output)
        memory_reads: list[str] = []
        memory_writes: dict[str, int] = {}
        branch_taken: bool | None = None
        step_type = "other"
        next_pc = pc + 1

        if instr.op == "PUSH_CONST":
            stack.append(int(instr.arg))
            step_type = "const"
        elif instr.op == "LOAD":
            name = str(instr.arg)
            memory_reads.append(name)
            if name not in env:
                halt_reason = f"name_error:{name}"
                break
            stack.append(env[name])
            step_type = "load"
        elif instr.op == "STORE":
            name = str(instr.arg)
            if not stack:
                halt_reason = "stack_underflow"
                break
            value = int(stack.pop())
            env[name] = value
            memory_writes[name] = value
            step_type = "store"
        elif instr.op in {"ADD", "SUB", "MUL"}:
            if len(stack) < 2:
                halt_reason = "stack_underflow"
                break
            right = int(stack.pop())
            left = int(stack.pop())
            stack.append(_eval_binary(instr.op, left, right))
            step_type = "arithmetic"
        elif instr.op.startswith("CMP_"):
            if len(stack) < 2:
                halt_reason = "stack_underflow"
                break
            right = int(stack.pop())
            left = int(stack.pop())
            stack.append(_eval_compare(instr.op, left, right))
            step_type = "compare"
        elif instr.op == "JMP_IF_FALSE":
            if not stack:
                halt_reason = "stack_underflow"
                break
            cond = int(stack.pop())
            branch_taken = cond == 0
            if branch_taken:
                next_pc = int(instr.arg)
            step_type = "control"
        elif instr.op == "JMP":
            branch_taken = True
            next_pc = int(instr.arg)
            step_type = "control"
        elif instr.op == "OUTPUT":
            if not stack:
                halt_reason = "stack_underflow"
                break
            output.append(int(stack.pop()))
            step_type = "output"
        elif instr.op == "HALT":
            step_type = "halt"
            env_after = _sorted_env(env)
            stack_after = list(stack)
            output_after = list(output)
            delta = _env_delta(env_before, env_after)
            trace.append(
                TraceEvent(
                    step=step,
                    pc=pc,
                    opcode=instr.op,
                    arg=instr.arg,
                    arg2=instr.arg2,
                    step_type=step_type,
                    stack_before=stack_before,
                    stack_after=stack_after,
                    env_before=env_before,
                    env_after=env_after,
                    env_delta=delta,
                    memory_reads=memory_reads,
                    memory_writes=memory_writes,
                    output_before=output_before,
                    output_after=output_after,
                    output_delta=output_after[len(output_before):],
                    branch_taken=branch_taken,
                    text=_event_text(instr.op, instr.arg, instr.arg2, stack_before, stack_after, delta, branch_taken, output_after[len(output_before):]),
                )
            )
            break
        else:
            raise ValueError(f"Unsupported instruction {instr.op!r}")

        env_after = _sorted_env(env)
        stack_after = list(stack)
        output_after = list(output)
        delta = _env_delta(env_before, env_after)
        trace.append(
            TraceEvent(
                step=step,
                pc=pc,
                opcode=instr.op,
                arg=instr.arg,
                arg2=instr.arg2,
                step_type=step_type,
                stack_before=stack_before,
                stack_after=stack_after,
                env_before=env_before,
                env_after=env_after,
                env_delta=delta,
                memory_reads=memory_reads,
                memory_writes=memory_writes,
                output_before=output_before,
                output_after=output_after,
                output_delta=output_after[len(output_before):],
                branch_taken=branch_taken,
                text=_event_text(instr.op, instr.arg, instr.arg2, stack_before, stack_after, delta, branch_taken, output_after[len(output_before):]),
            )
        )
        pc = next_pc
        step += 1
    else:
        halt_reason = "pc_out_of_range"

    return trace, _sorted_env(env), list(output), halt_reason


def capture_python_runtime_trace(python_source: str) -> tuple[list[dict[str, object]], list[int], dict[str, int]]:
    runtime_events: list[dict[str, object]] = []
    captured_output: list[int] = []

    def trace_print(*values: object) -> None:
        if len(values) == 1 and isinstance(values[0], int):
            captured_output.append(int(values[0]))
        else:
            captured_output.append(int(values[0]))  # type: ignore[arg-type]

    namespace: dict[str, object] = {"print": trace_print, "__builtins__": {}}
    wrapped_lines = ["def __trace_program__():"] + [f"    {line}" for line in python_source.splitlines()]
    wrapped_source = "\n".join(wrapped_lines) + "\n"
    exec(compile(wrapped_source, "<trace_program>", "exec"), namespace, namespace)
    fn = namespace["__trace_program__"]
    if not callable(fn):
        raise TypeError("Generated trace function is not callable")

    def tracer(frame, event, arg):  # type: ignore[no-untyped-def]
        if frame.f_code.co_name != "__trace_program__":
            return tracer
        if event in {"line", "return", "exception"}:
            locals_view = {
                key: int(value)
                for key, value in frame.f_locals.items()
                if not key.startswith("__") and isinstance(value, int)
            }
            payload: dict[str, object] = {
                "event": event,
                "line": int(frame.f_lineno),
                "locals": _sorted_env(locals_view),
            }
            if event == "return" and isinstance(arg, int):
                payload["return"] = int(arg)
            if event == "exception" and isinstance(arg, tuple) and arg:
                payload["exception"] = getattr(arg[0], "__name__", str(arg[0]))
            runtime_events.append(payload)
        return tracer

    prev = sys.gettrace()
    try:
        sys.settrace(tracer)
        fn()
    finally:
        sys.settrace(prev)

    final_locals = runtime_events[-1]["locals"] if runtime_events else {}
    return runtime_events, captured_output, dict(final_locals)


def _const_expr(builder: ExampleBuilder) -> ConstExpr:
    return ConstExpr(builder.rng.randint(builder.config.const_min, builder.config.const_max))


def _var_expr(available_vars: list[str], builder: ExampleBuilder) -> VarExpr:
    return VarExpr(builder.rng.choice(available_vars))


def _expr(builder: ExampleBuilder, available_vars: list[str], depth: int) -> Expr:
    if depth <= 0 or not available_vars:
        if available_vars and builder.rng.random() < 0.45:
            return _var_expr(available_vars, builder)
        return _const_expr(builder)
    mode = builder.rng.random()
    if mode < 0.30:
        return _const_expr(builder)
    if mode < 0.60 and available_vars:
        return _var_expr(available_vars, builder)
    left = _expr(builder, available_vars, depth - 1)
    right = _expr(builder, available_vars, depth - 1)
    return BinOpExpr(builder.rng.choice(["+", "-", "*"]), left, right)


def _cond(builder: ExampleBuilder, available_vars: list[str], depth: int) -> CompareExpr:
    left = _expr(builder, available_vars, max(depth - 1, 0))
    right = _expr(builder, available_vars, max(depth - 1, 0))
    return CompareExpr(builder.rng.choice(["<", "<=", ">", ">=", "==", "!="]), left, right)


def _make_initial_assignments(builder: ExampleBuilder, count: int) -> list[AssignStmt]:
    names = list(builder.config.variables)
    builder.rng.shuffle(names)
    body: list[AssignStmt] = []
    for name in names[:count]:
        body.append(AssignStmt(name, _const_expr(builder)))
    return body


def _make_branch_program(builder: ExampleBuilder) -> Program:
    init = _make_initial_assignments(builder, 2)
    available = [stmt.target for stmt in init]
    target = builder.rng.choice(list(builder.config.variables[2:5]))
    then_stmt = AssignStmt(target, _expr(builder, available, 1))
    else_stmt = AssignStmt(target, _expr(builder, available, 1))
    body: list[Stmt] = list(init)
    body.append(IfStmt(_cond(builder, available, 1), (then_stmt,), (else_stmt,)))
    body.append(EmitStmt(VarExpr(target)))
    return Program(tuple(body))


def _make_loop_program(builder: ExampleBuilder) -> Program:
    counter = builder.rng.choice(list(builder.config.variables[:3]))
    acc = builder.rng.choice([name for name in builder.config.variables if name != counter])
    start = builder.rng.randint(builder.config.loop_counter_min, builder.config.loop_counter_max)
    step = builder.rng.randint(1, 3)
    loop_body: list[Stmt] = [
        AssignStmt(acc, BinOpExpr("+", VarExpr(acc), VarExpr(counter))),
        AssignStmt(counter, BinOpExpr("-", VarExpr(counter), ConstExpr(step))),
    ]
    body: list[Stmt] = [
        AssignStmt(counter, ConstExpr(start)),
        AssignStmt(acc, ConstExpr(builder.rng.randint(0, 3))),
        WhileStmt(CompareExpr(">", VarExpr(counter), ConstExpr(0)), tuple(loop_body)),
        EmitStmt(VarExpr(acc)),
    ]
    return Program(tuple(body))


def _make_nested_program(builder: ExampleBuilder) -> Program:
    names = list(builder.config.variables)
    builder.rng.shuffle(names)
    outer = names[0]
    inner = names[1]
    acc = names[2]
    probe = names[3]
    # Keep nested traces short enough for the default smoke budget while still
    # exercising both nested control flow forms.
    start = 2
    inner_init = 1
    threshold = builder.rng.randint(1, max(start - 1, 1))
    then_probe = AssignStmt(probe, BinOpExpr("+", VarExpr(acc), VarExpr(outer)))
    else_probe = AssignStmt(probe, BinOpExpr("+", VarExpr(acc), ConstExpr(builder.rng.randint(1, 3))))
    nested_if = IfStmt(
        CompareExpr(">", VarExpr(outer), ConstExpr(threshold)),
        (then_probe,),
        (else_probe,),
    )
    inner_loop = WhileStmt(
        CompareExpr(">", VarExpr(inner), ConstExpr(0)),
        (
            AssignStmt(acc, BinOpExpr("+", VarExpr(acc), VarExpr(probe))),
            AssignStmt(inner, BinOpExpr("-", VarExpr(inner), ConstExpr(1))),
        ),
    )
    outer_loop = WhileStmt(
        CompareExpr(">", VarExpr(outer), ConstExpr(0)),
        (
            AssignStmt(inner, ConstExpr(inner_init)),
            nested_if,
            inner_loop,
            AssignStmt(outer, BinOpExpr("-", VarExpr(outer), ConstExpr(1))),
        ),
    )
    body: list[Stmt] = [
        AssignStmt(outer, ConstExpr(start)),
        AssignStmt(inner, ConstExpr(0)),
        AssignStmt(acc, ConstExpr(builder.rng.randint(0, 3))),
        AssignStmt(probe, ConstExpr(0)),
        outer_loop,
        EmitStmt(VarExpr(acc)),
    ]
    return Program(tuple(body))


def _make_straight_program(builder: ExampleBuilder) -> Program:
    init = _make_initial_assignments(builder, builder.rng.randint(2, 4))
    available = [stmt.target for stmt in init]
    body: list[Stmt] = list(init)
    extra_count = builder.rng.randint(1, max(builder.config.max_statements - len(body) - 1, 1))
    free_vars = [name for name in builder.config.variables if name not in available]
    for _ in range(extra_count):
        expr_available = list(available)
        if free_vars and builder.rng.random() < 0.5:
            target = free_vars.pop(0)
        else:
            target = builder.rng.choice(available)
        body.append(AssignStmt(target, _expr(builder, expr_available, builder.config.max_depth)))
        if target not in available:
            available.append(target)
    body.append(EmitStmt(_expr(builder, available, 1)))
    return Program(tuple(body))


def _sample_mixed_family(builder: ExampleBuilder) -> ConcreteTaskFamily:
    default_families: tuple[ConcreteTaskFamily, ...] = ("straight", "branch", "loop", "nested")
    mixture = builder.config.task_family_mixture
    if not mixture:
        return builder.rng.choice(list(default_families))
    families: list[ConcreteTaskFamily] = []
    weights: list[float] = []
    for name, weight in mixture:
        if name not in default_families:
            continue
        if float(weight) <= 0.0:
            continue
        families.append(name)  # type: ignore[arg-type]
        weights.append(float(weight))
    if not families:
        return builder.rng.choice(list(default_families))
    return builder.rng.choices(families, weights=weights, k=1)[0]


def generate_program(builder: ExampleBuilder) -> tuple[str, Program]:
    family = builder.config.task_family
    if family == "mixed":
        family = _sample_mixed_family(builder)
    if family == "straight":
        return family, _make_straight_program(builder)
    if family == "branch":
        return family, _make_branch_program(builder)
    if family == "loop":
        return family, _make_loop_program(builder)
    if family == "nested":
        return family, _make_nested_program(builder)
    raise ValueError(f"Unsupported task family {family!r}")


def build_example(builder: ExampleBuilder) -> dict[str, object]:
    resolved_family, program = generate_program(builder)
    instructions = compile_program(program)
    trace, final_env, output, halt_reason = execute_program(instructions, max_steps=builder.config.max_trace_steps)
    if halt_reason != "halt":
        raise RuntimeError(f"Generated program did not halt cleanly: {halt_reason}")

    source_text = program_to_source(program)
    python_source = program_to_python_source(program)
    python_runtime_trace, python_output, python_final_locals = capture_python_runtime_trace(python_source)
    if python_output != output:
        raise RuntimeError(
            f"Python runtime output mismatch: vm={output} python={python_output} source={python_source!r}"
        )
    if python_final_locals != final_env:
        raise RuntimeError(
            f"Python runtime locals mismatch: vm={final_env} python={python_final_locals} source={python_source!r}"
        )
    ast_payload = program_to_ast(program)
    trace_payload = [asdict(event) for event in trace]
    python_runtime_trace_text = [
        f"event={event['event']} line={event['line']} locals={event['locals']}"
        + (f" return={event['return']}" if "return" in event else "")
        + (f" exception={event['exception']}" if "exception" in event else "")
        for event in python_runtime_trace
    ]
    return {
        "schema_version": "execution_trace_v1",
        "example_id": builder.fresh_id(),
        "task_family": resolved_family,
        "requested_task_family": builder.config.task_family,
        "source": {
            "language": "tiny_exec_v1",
            "text": source_text,
            "python_like_text": python_source,
            "ast": ast_payload,
            "python_ast_dump": ast.dump(ast.parse(python_source), indent=2),
        },
        "ir": {
            "vm": "stack_vm_v1",
            "instructions": [instr.to_dict() for instr in instructions],
            "opcode_text": [
                " ".join(
                    str(part)
                    for part in [instr.op, instr.arg, instr.arg2]
                    if part is not None
                )
                for instr in instructions
            ],
        },
        "trace": trace_payload,
        "views": {
            "trace_text": [event["text"] for event in trace_payload],
            "memory_trace_text": [
                f"step={event['step']} pc={event['pc']} reads={event['memory_reads']} writes={event['memory_writes']} "
                f"env_delta={event['env_delta']} output_delta={event['output_delta']}"
                for event in trace_payload
            ],
            "python_runtime_trace_text": python_runtime_trace_text,
        },
        "python_runtime": {
            "events": python_runtime_trace,
            "output": python_output,
            "final_locals": python_final_locals,
        },
        "final": {
            "env": final_env,
            "output": output,
            "halt_reason": halt_reason,
        },
    }


def generate_examples(
    count: int,
    *,
    seed: int = 0,
    max_statements: int = 6,
    max_depth: int = 2,
    max_trace_steps: int = 128,
    task_family: TaskFamily = "mixed",
    task_family_mixture: tuple[tuple[str, float], ...] = (),
) -> list[dict[str, object]]:
    config = GenerationConfig(
        seed=seed,
        max_statements=max_statements,
        max_depth=max_depth,
        max_trace_steps=max_trace_steps,
        task_family=task_family,
        task_family_mixture=task_family_mixture,
    )
    builder = ExampleBuilder(random.Random(seed), config)
    return [build_example(builder) for _ in range(count)]


def example_to_jsonl(example: dict[str, object]) -> str:
    return json.dumps(example, sort_keys=True)
