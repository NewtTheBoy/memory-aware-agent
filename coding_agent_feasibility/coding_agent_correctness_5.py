#!/usr/bin/env python3
"""Code investigation feasibility workload with optional external case datasets.

LLM: local llama.cpp OpenAI-compatible chat API.
Tools: real repository grep/read and safe, allowlisted pytest commands.
No artificial sleeps and no model-generated shell commands.
This is a small feasibility test, not a standard coding-agent benchmark.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def http_json(url: str, *, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    with urlopen(Request(url, headers=headers, data=data), timeout=timeout) as resp:
        return json.load(resp)


def parse_json_reply(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        hit = re.search(r"\{.*\}", raw, flags=re.S)
        if not hit:
            raise ValueError(f"Expected a JSON object, got {raw[:180]!r}")
        obj = json.loads(hit.group())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object, got {type(obj).__name__}")
    return obj


class CodingPilot:
    def __init__(self, repo: Path, api: str, python: str, max_llm: int, test_timeout: int, max_followups: int = 2):
        self.repo = repo.resolve()
        self.api = api.rstrip("/")
        self.python = python
        self.sem = asyncio.Semaphore(max_llm)
        self.timeout = test_timeout
        self.max_followups = max_followups
        self.investigation_profile = "basic"
        self.profile_instructions = ""
        self.events: list[dict] = []
        self.started = time.perf_counter()
        self.model = ""
        self.question = (
            "Investigate how Click handles command-line options. Identify the relevant "
            "source code and tests, run a focused test file, then decide based on real "
            "results whether to inspect the test file, run the full test suite, or finish. "
            "Report the evidence and limitations. Do not edit files."
        )

    def now(self):
        return round(time.perf_counter() - self.started, 6)

    def record(self, ev):
        ev["end_wall_epoch_s"] = time.time()
        ev["end_s"] = self.now()
        ev["duration_s"] = round(ev["end_s"] - ev["start_s"], 6)
        self.events.append(ev)
        status = "ERROR" if "error" in ev else "CANCEL" if ev.get("cancelled") else "OK"
        print(f"[{ev['id']:>3}] {ev['kind']:<14} {status:<5} "
              f"{ev['duration_s']:>7.3f}s  {ev.get('detail','')}", flush=True)

    async def identify(self):
        response = await asyncio.to_thread(http_json, f"{self.api}/models", timeout=10)
        self.model = response["data"][0]["id"]
        print(f"Model: {self.model}\nRepository: {self.repo}", flush=True)

    async def llm(self, sid: str, deps: list[str], messages: list[dict], max_tokens=240):
        ready = self.now()
        async with self.sem:
            start = self.now()
            ev = {"id": sid, "kind": "llm", "depends_on": deps,
                  "start_wall_epoch_s": time.time(),
                  "ready_s": ready, "start_s": start,
                  "queue_s": round(start - ready, 6)}
            try:
                response = await asyncio.to_thread(
                    http_json, f"{self.api}/chat/completions",
                    payload={"model": self.model, "messages": messages,
                             "temperature": 0, "max_tokens": max_tokens},
                    timeout=120,
                )
                content = response["choices"][0]["message"].get("content") or ""
                ev["output"] = content
                ev["usage"] = response.get("usage", {})
                return content
            except Exception as exc:
                ev["error"] = repr(exc)
                raise
            except asyncio.CancelledError:
                ev["cancelled"] = True
                raise
            finally:
                self.record(ev)

    async def tool(self, sid: str, deps: list[str], name: str, fn, *, detail=""):
        start = self.now()
        ev = {"id": sid, "kind": name, "depends_on": deps,
              "start_wall_epoch_s": time.time(),
              "ready_s": start, "start_s": start, "detail": detail}
        try:
            result = await asyncio.to_thread(fn)
            ev["result"] = result
            return result
        except Exception as exc:
            ev["error"] = repr(exc)
            raise
        finally:
            self.record(ev)

    def grep_files(self, root: str, term: str):
        # Restrict to the trusted project source/test trees; no shell commands.
        base = self.repo / root
        regex = re.compile(re.escape(term[:70]), re.I)
        files = []
        for p in sorted(base.rglob("*.py")):
            try:
                lines = p.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            matches = [(i, line.strip()[:110]) for i, line in enumerate(lines, 1)
                       if regex.search(line)]
            # Search filenames as well as content: e.g. "test_options" should
            # find tests/test_options.py even when the exact word is absent inside.
            if matches or regex.search(p.name):
                files.append({"path": p.relative_to(self.repo).as_posix(),
                              "count": len(matches), "samples": matches[:2]})
        files.sort(key=lambda v: -v["count"])
        return files[:8]

    def rank_files_by_question(self, root: str, limit: int = 8) -> list[dict]:
        """Repository-only lexical fallback for failed literal grep.

        Score real module names, AST definitions, and small line samples using
        question words. Never read expected_source/symbol/test labels. These
        candidates are NOT used as an exact-match grader or an oracle.
        """
        def stems(values: set[str]) -> set[str]:
            return {value[:5] for value in values}

        target = stems(self._navigation_terms(self.question))
        ranked = []
        for path in sorted((self.repo / root).rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, UnicodeError, SyntaxError):
                continue
            rel = path.relative_to(self.repo).as_posix()
            file_terms = stems(self._navigation_terms(path.stem.replace("_", " ")))
            file_score = 8 * len(file_terms & target)
            defs = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    hint = (ast.get_docstring(node) or "").splitlines()
                    score = self._rank_symbol(node.name, hint[0] if hint else "", self.question)
                    defs.append((score, node.name))
            top_score = max((score for score, _ in defs), default=0)
            examples = [name for score, name in sorted(defs, reverse=True)
                        if score > 0][:3]
            # A small content signal (no full-file keyword-frequency bias).
            content_terms = stems(self._navigation_terms(" ".join(source.splitlines()[:120])))
            score = file_score + 3 * top_score + min(4, len(content_terms & target))
            if score:
                ranked.append({"path": rel, "count": score,
                               "samples": examples,
                               "discovery": "question_lexical_fallback"})
        return sorted(ranked, key=lambda item: (-item["count"], item["path"]))[:limit]

    def relevant_source_excerpt(self, file_path: Path, search_term: str, budget: int = 14000):
        """Bounded, line-numbered source excerpts around the AGENT-selected grep term.

        Avoid only reading the start of a long source file. We do not use ground
        truth labels to select evidence. This is a tool view, not an oracle.
        """
        lines = file_path.read_text(encoding="utf-8").splitlines()
        hits = [i for i, line in enumerate(lines)
                if search_term.lower() in line.lower()]
        if not hits:
            hits = [0]
        # Sample across the file for common keywords, and include the first
        # matching definition when possible. Limit the total observation size.
        selected = []
        if len(hits) <= 9:
            selected = hits
        else:
            selected = [hits[round(j * (len(hits)-1) / 8)] for j in range(9)]
        blocks = []
        for hit in selected:
            low, high = max(0, hit - 14), min(len(lines), hit + 36)
            excerpt = "\n".join(f"{i+1}: {lines[i]}" for i in range(low, high))
            blocks.append(f"--- source lines {low+1}-{high} ---\n{excerpt}")
        return (f"File: {file_path.relative_to(self.repo).as_posix()} "
                f"({len(lines)} lines); grep term: {search_term}\n" +
                "\n".join(blocks))[:budget]

    @staticmethod
    def _navigation_terms(text: str) -> set[str]:
        # Generic search relevance; no task labels, expected paths, or answers.
        tokens = re.findall(r"[a-z]+|[0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
        ignored = {"the", "and", "into", "from", "with", "this", "that", "when",
                   "where", "file", "test", "tests", "source", "code", "click",
                   "command", "commands", "related", "locate", "implementation",
                   "investigate", "how", "its", "does", "runs", "using", "which",
                   "return", "value", "result", "outputs", "output", "handles"}
        return {t for t in tokens if len(t) >= 3 and t not in ignored}

    def _rank_symbol(self, symbol: str, hint: str, question: str) -> int:
        terms = self._navigation_terms(question)
        name = self._navigation_terms(symbol.replace("_", " "))
        description = self._navigation_terms(hint)
        # Match light inflections ("resolves" vs "resolve") without label knowledge.
        stem = lambda x: x[:5] if len(x) >= 5 else x
        terms_stem = {stem(x) for x in terms}
        name_stem = {stem(x) for x in name}
        description_stem = {stem(x) for x in description}
        score = 5 * len(terms_stem & name_stem) + len(terms_stem & description_stem)
        return score

    @staticmethod
    def _source_symbols(path: Path) -> list[dict]:
        contents = path.read_text(encoding="utf-8")
        tree = ast.parse(contents, filename=str(path))
        collected = []

        def add(node, qualified):
            doc = (ast.get_docstring(node) or "").strip().splitlines()
            hint = doc[0][:125].strip() if doc else ""
            collected.append({"symbol": qualified, "line": node.lineno,
                              "end_line": node.end_lineno, "hint": hint,
                              "_node": node})

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(node, node.name)
            elif isinstance(node, ast.ClassDef):
                # Class itself is not a callable implementation method.
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        add(child, f"{node.name}.{child.name}")
        return collected

    def source_symbol_index(self, path: Path, max_entries: int = 65) -> dict:
        symbols = self._source_symbols(path)
        # Relevance ranking reduces context consumption in huge modules.
        ranked = sorted(symbols, key=lambda o: (-self._rank_symbol(o["symbol"], o["hint"], self.question), o["line"]))
        shown = ranked[:max_entries]
        return {"path": path.relative_to(self.repo).as_posix(),
                "total_symbols": len(symbols), "shown": len(shown),
                "symbols": [{"symbol": s["symbol"], "line": s["line"],
                             "hint": s["hint"][:90]} for s in shown],
                # All real candidates remain selectable, even in core.py.
                "other_allowed_symbols": [s["symbol"] for s in ranked[max_entries:]]}

    def read_source_symbol(self, path: Path, selected: str, max_chars: int = 11500) -> str:
        symbols = {s["symbol"]: s for s in self._source_symbols(path)}
        if selected not in symbols:
            raise ValueError(f"Symbol {selected!r} not present in {path}")
        item = symbols[selected]
        lines = path.read_text(encoding="utf-8").splitlines()
        low = item["line"] - 1
        high = min(len(lines), item["end_line"])
        # Include enclosing class line for qualified class methods.
        if "." in selected:
            class_name = selected.split(".", 1)[0]
            for i in range(low - 1, -1, -1):
                if re.match(r"class\s+" + re.escape(class_name) + r"\b", lines[i]):
                    header = f"Enclosing class at line {i+1}: {lines[i].strip()}\n"
                    break
            else:
                header = ""
        else:
            header = ""
        body = "\n".join(f"{i+1}: {lines[i]}" for i in range(low, high))
        if len(body) > max_chars:
            lead, tail = max_chars * 3 // 4, max_chars // 4
            body = body[:lead] + "\n... [middle of selected symbol omitted] ...\n" + body[-tail:]
        return (f"File: {path.relative_to(self.repo).as_posix()}\n"
                f"Selected symbol: {selected} (lines {item['line']}-{item['end_line']})\n"
                + header + body)

    def find_related_symbols(self, path: Path, selected: str, limit: int = 12) -> list[dict]:
        """Find directly called source helpers, including local relative imports.

        The model chooses a helper from actual AST-derived call sites; no
        benchmark label or guessed filename enters this process.
        """
        current = {o["symbol"]: o for o in self._source_symbols(path)}
        if selected not in current:
            return []
        class_name = selected.split(".", 1)[0] if "." in selected else None
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                target = (self.repo / "src/click" / (node.module.replace(".", "/") + ".py")).resolve()
                if target.is_file() and target.is_relative_to(self.repo):
                    for name in node.names:
                        imported[name.asname or name.name] = (target, name.name)

        found = {}
        for call in ast.walk(current[selected]["_node"]):
            if not isinstance(call, ast.Call):
                continue
            targets = []
            fn = call.func
            if isinstance(fn, ast.Name):
                if fn.id in current:
                    targets.append((path, fn.id, "local function call"))
                if fn.id in imported:
                    imported_path, imported_symbol = imported[fn.id]
                    targets.append((imported_path, imported_symbol, "relative-import call"))
            elif isinstance(fn, ast.Attribute):
                if isinstance(fn.value, ast.Name) and fn.value.id == "self" and class_name:
                    qualified = f"{class_name}.{fn.attr}"
                    if qualified in current:
                        targets.append((path, qualified, "self.method call"))
                elif isinstance(fn.value, ast.Name):
                    qualified = f"{fn.value.id}.{fn.attr}"
                    if qualified in current:
                        targets.append((path, qualified, "class method call"))
            for target_path, symbol, reason in targets:
                if target_path == path and symbol == selected:
                    continue
                try:
                    target_index = {o["symbol"]: o for o in self._source_symbols(target_path)}
                except (OSError, UnicodeError, SyntaxError):
                    continue
                if symbol not in target_index:
                    continue
                rel = target_path.relative_to(self.repo).as_posix()
                found[(rel, symbol)] = {"path": rel, "symbol": symbol,
                                        "line": target_index[symbol]["line"],
                                        "reason": reason}
        return sorted(found.values(), key=lambda item: (
            -self._rank_symbol(item["symbol"], "", self.question),
            item["path"], item["symbol"]))[:limit]

    def read_test_evidence(self, path: Path, limit: int = 3, budget: int = 9000) -> str:
        """Read genuine test functions, ranked by question relevance.

        The most promising test bodies are shown with line numbers, not only
        the first lines of a potentially long test module.
        """
        contents = path.read_text(encoding="utf-8")
        lines = contents.splitlines()
        tree = ast.parse(contents, filename=str(path))
        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                funcs.append(node)
        scored = sorted(funcs, key=lambda node: (
            -self._rank_symbol(node.name, ast.get_docstring(node) or "", self.question),
            node.lineno))[:limit]
        output = [f"File: {path.relative_to(self.repo).as_posix()}; total test functions={len(funcs)}"]
        for node in scored:
            low, high = node.lineno - 1, min(node.end_lineno, node.lineno + 45)
            output.append(f"Test: {node.name} (lines {node.lineno}-{node.end_lineno})\n" +
                          "\n".join(f"{idx + 1}: {lines[idx]}" for idx in range(low, high)))
        return "\n\n".join(output)[:budget]

    @staticmethod
    def _test_functions(path: Path) -> list[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError, OSError):
            return []
        funcs = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                funcs.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                        funcs.append(f"{node.name}.{child.name}")
        return funcs

    def index_test_candidates(self, files: list[dict]) -> dict:
        # All candidates originate from actual grep/list tools, not gold labels.
        indexed = []
        for entry in files:
            path = self.safe_path(entry["path"], [entry["path"]])
            names = self._test_functions(path)
            names_sorted = sorted(names, key=lambda name: (-self._rank_symbol(name, "", self.question), name))
            indexed.append({"path": entry["path"], "test_count": len(names),
                            "example_tests": names_sorted[:4],
                            "grep_count": entry.get("count", 0)})
        return {"file_count": len(indexed), "files": indexed}

    def compact_test_index(self, index: dict, preview_limit: int = 12) -> dict:
        """Keep *all* actual paths available, but bound function-name context.

        Ranking uses only the current task and actual repository test names;
        evaluator labels are never consulted. Any unpreviewed path remains valid.
        """
        def relevance(item: dict) -> int:
            module_name = Path(item["path"]).stem
            module_score = self._rank_symbol(module_name.replace("_", " "), "", self.question)
            function_score = max((self._rank_symbol(name.replace("_", " "), "", self.question)
                                  for name in item["example_tests"]), default=0)
            return 2 * module_score + function_score

        ranked = sorted(index["files"], key=lambda item: (
            -relevance(item), -item.get("grep_count", 0), item["path"]))
        return {
            "total_count": len(ranked),
            "detailed_candidates": ranked[:preview_limit],
            "other_allowed_paths": [item["path"] for item in ranked[preview_limit:]],
        }

    async def choose_valid_test_path(self, messages: list[dict],
                                     options: list[str]) -> tuple[str, str]:
        """Let the agent correct an invalid B2 choice without fabricating a path.

        On an invalid choice, append the failed reply and concrete validation
        feedback to the ongoing conversation, and re-ask with the *same* real
        allowlist. Each attempt is an ordinary traced LLM step.
        """
        step, deps = "B2", ["B1I"]
        for retry in range(3):  # initial choice + at most two recovery calls
            raw = await self.llm(step, deps, messages, max_tokens=130)
            try:
                obj = parse_json_reply(raw)
                requested = obj.get("path")
                if not isinstance(requested, str):
                    raise ValueError("The reply must have a string 'path' field")
                self.safe_path(requested, options)
            except (ValueError, TypeError) as exc:
                if retry == 2:
                    raise ValueError(
                        f"B2 selected no valid test path after 3 attempts; "
                        f"last reply={raw[:180]!r}; error={exc}"
                    ) from exc
                print(f"Invalid {step} selection; requesting a corrected path: {exc}",
                      flush=True)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    "VALIDATION ERROR: your previous test path is not a permitted "
                    "repository file (or was not valid JSON). "
                    f"Details: {exc}.\n"
                    f"Exact permitted paths: {json.dumps(options)}\n"
                    'Choose ONE of these paths verbatim. Reply only {"path":"..."}. '
                    "Do not construct a new filename or repeat the invalid choice."})
                deps = [step]
                step = f"B2R{retry + 1}"
                continue
            messages.append({"role": "assistant", "content": raw})
            return requested, step
        raise AssertionError("unreachable")

    def safe_path(self, requested: str, allowed: list[str]):
        # No absolute paths, traversal or arbitrary model-selected files.
        if requested not in allowed:
            raise ValueError(f"Model selected unexpected path {requested!r}; valid={allowed}")
        p = (self.repo / requested).resolve()
        if not p.is_relative_to(self.repo) or not p.is_file():
            raise ValueError(f"Unsafe or missing path: {requested}")
        return p

    async def pytest(self, sid: str, deps: list[str], test_path: str | None):
        # Fixed argv: model cannot generate arbitrary code or shell commands.
        argv = [self.python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        if test_path is not None:
            self.safe_path(test_path, [test_path] if test_path.startswith("tests/")
                           and test_path.endswith(".py") else [])
            argv.append(test_path)
        start = self.now()
        ev = {"id": sid, "kind": "pytest", "depends_on": deps,
              "start_wall_epoch_s": time.time(),
              "start_s": start, "ready_s": start, "command": argv}
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(*argv, cwd=self.repo,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
                ev["exit_code"] = proc.returncode
                ev["output"] = stdout.decode(errors="replace")[-5000:]
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                ev["exit_code"] = None
                ev["timed_out"] = True
                ev["output"] = f"Timed out after {self.timeout}s"
            return {"exit_code": ev["exit_code"], "output": ev["output"],
                    "timed_out": ev.get("timed_out", False)}
        except Exception as exc:
            ev["error"] = repr(exc)
            raise
        finally:
            self.record(ev)

    def list_source_candidates(self):
        """List repository source modules, without evaluator gold labels."""
        paths = sorted(p.relative_to(self.repo).as_posix()
                       for p in (self.repo / "src/click").rglob("*.py"))
        return [{"path": path, "count": 0, "samples": [],
                 "discovery": "source_file_listing"} for path in paths]

    async def source_branch(self, source_term: str):
        files = await self.tool("A1", ["P0"], "grep_source",
                 lambda: self.grep_files("src/click", source_term), detail=source_term)
        search_dep = "A1"
        if not files:
            files = await self.tool("A1Q", ["A1"], "rank_source_by_question",
                                    lambda: self.rank_files_by_question("src/click"),
                                    detail="lexical fallback after zero literal hits")
            search_dep = "A1Q"
        if (not files or len(files) >= 8 or
                any(x.get("discovery") == "question_lexical_fallback" for x in files)):
            # The grep tool caps results at 8. Common keywords such as the
            # project name can hide the relevant module. Expose *all* module
            # filenames in a second tool step; do not use benchmark labels.
            listed = await self.tool("A1L", [search_dep], "list_source_files",
                                     self.list_source_candidates,
                                     detail="expand broad/empty source search")
            existing = {x["path"] for x in files}
            files += [x for x in listed if x["path"] not in existing]
            search_dep = "A1L"
        if not files:
            raise RuntimeError("Repository has no discoverable source modules")
        options = [x["path"] for x in files]
        messages = [
            {"role": "system", "content": "You are a code investigation agent. Respond with valid JSON only."},
            {"role": "user", "content": f"Task: {self.question}\nSearch term: {source_term}\n"
                f"Source discovery results: {json.dumps(files)}\nChoose ONE relevant source file "
                'from the list and return {"path":"..."}.'},
        ]
        picked = await self.llm("A2", [search_dep], messages)
        path = str(parse_json_reply(picked)["path"])
        file_path = self.safe_path(path, options)
        messages.append({"role": "assistant", "content": picked})
        index = await self.tool("A3I", ["A2"], "index_source_symbols",
                                lambda: self.source_symbol_index(file_path), detail=path)
        symbols = [s["symbol"] for s in index["symbols"]] + index["other_allowed_symbols"]
        symbol_prompt = [
            {"role": "system", "content": "You are a code navigation agent. Reply with a JSON object only."},
            {"role": "user", "content":
                f"Task: {self.question}\n"
                f"Selected source file: {path}\n"
                f"Actual source function/method index: {json.dumps(index)}\n"
                'Choose the single implementation function or method most relevant to this task. '
                'Use an exact symbol from the preview OR other_allowed_symbols. '
                'Return {"symbol":"..."}. Do not invent symbols.'},
        ]
        symbol_response = await self.llm("A3S", ["A3I"], symbol_prompt, max_tokens=100)
        symbol = str(parse_json_reply(symbol_response).get("symbol", "")).strip()
        if symbol not in symbols:
            # One generic correction opportunity, never a forced gold answer.
            correction = await self.llm("A3R", ["A3S"], symbol_prompt + [
                {"role": "assistant", "content": symbol_response},
                {"role": "user", "content":
                    f"That symbol {symbol!r} is not in the source index. "
                    f"Choose exactly one from: {json.dumps(symbols)}. "
                    'Return {"symbol":"..."} only.'},
            ], max_tokens=110)
            symbol = str(parse_json_reply(correction).get("symbol", "")).strip()
            if symbol not in symbols:
                raise ValueError(f"Agent selected missing source symbol {symbol!r}")
            read_dep = "A3R"
        else:
            read_dep = "A3S"
        content = await self.tool("A3", [read_dep], "read_source_symbol",
                                  lambda: self.read_source_symbol(file_path, symbol),
                                  detail=f"{path}:{symbol}")
        helper_available = False
        helper_selected = None
        helper_content = ""
        analysis_dep = "A3"
        if self.investigation_profile in ("call_trace", "deep"):
            candidates = await self.tool(
                "A3C", ["A3"], "find_related_symbols",
                lambda: self.find_related_symbols(file_path, symbol),
                detail=f"direct callees of {symbol}")
            analysis_dep = "A3C"
            # Defense in depth: the current primary symbol must never be a
            # candidate for a *related* helper. find_related_symbols also
            # filters self-calls, but keep this boundary local to selection.
            primary = (path, symbol)
            candidates = [item for item in candidates
                          if (item["path"], item["symbol"]) != primary]
            helper_available = bool(candidates)
            if candidates:
                allowed = {(item["path"], item["symbol"]) for item in candidates}
                helper_prompt = [
                    {"role": "system", "content":
                        "You are a source tracing agent. Choose ONLY from the "
                        "listed directly called helpers. Return valid JSON only. "
                        "Do not select the primary symbol itself."},
                    {"role": "user", "content":
                        f"Task: {self.question}\n"
                        f"Primary implementation (NOT selectable): {path}:{symbol}\n"
                        f"Allowed direct helper candidates: {json.dumps(candidates)}\n"
                        'Pick one listed helper that best explains the primary '
                        'implementation. Return {"path":"...","symbol":"..."} '
                        'using a candidate pair verbatim. Do not invent a path '
                        'or return the primary implementation.'},
                ]
                step, deps = "A3H", ["A3C"]
                chosen = None
                for attempt in range(2):  # One correction, never an oracle fallback.
                    raw_helper = await self.llm(step, deps, helper_prompt, max_tokens=130)
                    try:
                        parsed = parse_json_reply(raw_helper)
                        choice = (parsed.get("path"), parsed.get("symbol"))
                        if not all(isinstance(v, str) for v in choice) or choice not in allowed:
                            raise ValueError(f"Helper pair {choice!r} is not in the candidate list")
                    except (ValueError, TypeError) as exc:
                        print(f"Invalid {step} helper selection: {exc}", flush=True)
                        analysis_dep = step
                        if attempt == 1:
                            print("A3H retry exhausted; no helper was inspected", flush=True)
                            break
                        helper_prompt.extend([
                            {"role": "assistant", "content": raw_helper},
                            {"role": "user", "content":
                                f"VALIDATION ERROR: {exc}. The primary symbol "
                                f"{path}:{symbol} is NOT a helper. "
                                f"Valid pairs ONLY: {json.dumps(candidates)}. "
                                'Select one of these exact path+symbol pairs. '
                                'Return only {"path":"...","symbol":"..."}.'},
                        ])
                        deps, step = [step], "A3HR1"
                        continue
                    chosen = choice
                    break
                if chosen is not None:
                    helper_path = self.safe_path(chosen[0], [o["path"] for o in candidates])
                    helper_content = await self.tool("A3R", [step], "read_related_helper",
                        lambda: self.read_source_symbol(helper_path, chosen[1], max_chars=7000),
                        detail=f"{chosen[0]}:{chosen[1]}")
                    helper_selected = {"path": chosen[0], "symbol": chosen[1]}
                    analysis_dep = "A3R"
        analysis = await self.llm("A4", [analysis_dep], [
            {"role": "system", "content":
                "You are a code investigation agent. Refer only to observed source evidence."},
            {"role": "user", "content":
                f"Task: {self.question}{self.profile_instructions}\n"
                f"Source symbol selected through navigation: {symbol}\n"
                f"TOOL RESULT read_source_symbol:\n{content}\n"
                + (f"TOOL RESULT related helper:\n{helper_content}\n" if helper_content else "")
                + "Explain what the selected primary symbol implements, with source line evidence. "
                + ("Explain the directly-called helper and why it matters. " if helper_content else "")
                + "If the inspected symbol does not fit the task, say so explicitly."},
        ], max_tokens=450 if helper_content else 350)
        return {"source_path": path, "source_symbol": symbol,
                "source_analysis": analysis,
                "related_helper_available": helper_available,
                "related_helper": helper_selected}

    def list_test_candidates(self):
        """Discover real test modules without relying on a possibly bad LLM keyword.

        No expected answer labels are used. This is an ordinary directory-listing
        fallback; the agent must still choose a file and run the real test.
        """
        paths = sorted(p.relative_to(self.repo).as_posix()
                       for p in (self.repo / "tests").rglob("test_*.py"))
        return [{"path": path, "count": 0, "samples": [],
                 "discovery": "test_file_listing"} for path in paths]

    async def test_branch(self, test_term: str):
        # Initial discovery is structured, but what follows the focused test is
        # intentionally NOT fixed ahead of time: the model decides each step
        # after observing the actual tool result. Only revealed steps get logged.
        files = await self.tool("B1", ["P0"], "grep_tests",
                 lambda: self.grep_files("tests", test_term), detail=test_term)
        search_dep = "B1"
        if not files:
            files = await self.tool("B1Q", ["B1"], "rank_tests_by_question",
                                    lambda: self.rank_files_by_question("tests"),
                                    detail="lexical fallback after zero literal hits")
            search_dep = "B1Q"
        if not files or any(x.get("discovery") == "question_lexical_fallback" for x in files):
            # A bad search term is NOT a reason to abort the whole task.
            # Expose repository test filenames through a real listing tool and
            # let the model pick from them; do not use evaluator gold labels.
            listed = await self.tool("B1L", [search_dep], "list_test_files",
                                     self.list_test_candidates,
                                     detail="complete test candidate allowlist")
            existing = {x["path"] for x in files}
            files += [x for x in listed if x["path"] not in existing]
            search_dep = "B1L"
            print(f"Test search {test_term!r} found no matches; "
                  f"listed {len(files)} real test modules instead.", flush=True)
        if not files:
            raise RuntimeError("Repository has no discoverable test_*.py modules")

        options = [x["path"] for x in files]
        test_index = await self.tool("B1I", [search_dep], "index_test_functions",
                                     lambda: self.index_test_candidates(files),
                                     detail=f"{len(files)} candidate modules")
        # Avoid flooding a small local model with four function names for
        # every test module. Show detailed evidence for the most relevant 12,
        # but retain *every* real module path as a legal candidate.
        test_view = self.compact_test_index(test_index)
        messages = [
            {"role": "system", "content":
                "You are a code investigation agent. Read tool results and make decisions. "
                "When asked for an action, respond with one valid JSON object only. "
                "For file selection, copy an exact path from the given list; "
                "never invent or autocomplete filenames."},
            {"role": "user", "content":
                f"Task: {self.question}\n"
                f"Actual test module and function index (compact view): "
                f"{json.dumps(test_view)}\n"
                'Choose the test module most relevant to this task based on module and '
                'test function names. All listed paths (including other_allowed_paths) '
                'are allowed. Return {"path":"..."} with an EXACT listed path.'},
        ]
        test_path, choice_step = await self.choose_valid_test_path(messages, options)
        targeted = await self.pytest("B3", [choice_step], test_path)

        # Test-evidence tasks must actually inspect test source; pytest PASS
        # alone is not evidence that a module tests the requested behavior.
        last_step = "B3"
        decision_history = []
        followup_actions = 0
        inspected_test = False
        test_evidence_analysis = None
        full_result = None
        status = "not_decided"
        if self.investigation_profile in ("test_evidence", "deep"):
            file_path = self.safe_path(test_path, options)
            evidence = await self.tool("B3I", ["B3"], "read_test_evidence",
                                       lambda: self.read_test_evidence(file_path),
                                       detail=test_path)
            test_evidence_analysis = await self.llm("B3E", ["B3I"], [
                {"role": "system", "content":
                    "You are evaluating TEST EVIDENCE, not whether pytest simply passed."},
                {"role": "user", "content":
                    f"Task: {self.question}\nActual test source excerpts:\n{evidence}\n"
                    "Identify one observed test function, what it checks, and whether it "
                    "supports the task. State limitations. Do not invent assertions."},
            ], max_tokens=240)
            inspected_test = True
            last_step = "B3E"
            messages.append({"role": "user", "content":
                f"TOOL RESULT focused pytest ({test_path}): {json.dumps(targeted)}\n"
                f"TOOL RESULT inspected test source:\n{evidence}\n"
                f"Test evidence assessment: {test_evidence_analysis}"})
        else:
            messages.append({"role": "user", "content":
                f"TOOL RESULT focused pytest ({test_path}):\n{json.dumps(targeted)}"})
        messages.append({"role": "user", "content":
            "Choose next action from actual evidence: "
            "run_full_tests (broad regression check), inspect_test_file "
            "(read the selected test source, if not done), or finish (enough evidence). "
            'Return JSON like {"action":"finish","reason":"..."}. '
            "Do not invent other actions. Avoid repeating completed actions."})

        # Allow up to max_followups tool actions. A finish decision can happen
        # immediately after the focused test (zero additional tools).
        for decision_idx in range(self.max_followups + 1):
            decision_step = f"D{decision_idx}"
            raw = await self.llm(decision_step, [last_step], messages, max_tokens=140)
            obj = parse_json_reply(raw)
            action = str(obj.get("action", "")).strip()
            allowed = {"finish"}
            if full_result is None and followup_actions < self.max_followups:
                allowed.add("run_full_tests")
            if not inspected_test and followup_actions < self.max_followups:
                allowed.add("inspect_test_file")
            if action not in allowed:
                raise ValueError(
                    f"Invalid/duplicate agent action {action!r}; permitted={sorted(allowed)}"
                )
            decision_history.append({"step": decision_step, "action": action,
                                     "reason": str(obj.get("reason", ""))})
            print(f"Decision {decision_step}: {action} "
                  f"({str(obj.get('reason', ''))[:120]})", flush=True)
            messages.append({"role": "assistant", "content": raw})
            if action == "finish":
                status = "agent_finished"
                last_step = decision_step
                break

            followup_actions += 1
            if action == "run_full_tests":
                tool_step = f"T_full_{followup_actions}"
                full_result = await self.pytest(tool_step, [decision_step], None)
                response = f"TOOL RESULT full pytest:\n{json.dumps(full_result)}"
            else:
                tool_step = f"T_read_{followup_actions}"
                file_path = self.safe_path(test_path, options)
                contents = await self.tool(
                    tool_step, [decision_step], "read_test_file",
                    lambda: file_path.read_text(encoding="utf-8")[:6500],
                    detail=test_path,
                )
                inspected_test = True
                response = f"TOOL RESULT read_test_file({test_path}):\n{contents}"
            last_step = tool_step
            messages.append({"role": "user", "content":
                response + "\nBased on this NEW tool result, choose your next action. "
                'Reply as JSON: {"action":"finish|run_full_tests|inspect_test_file",'
                '"reason":"..."}. Only choose a tool not already executed.'})
        else:
            status = "decision_budget_exhausted"

        messages.append({"role": "user", "content":
            "Summarize the observed test results and limits, based strictly on "
            "the tool outputs above. Do not claim tests ran if they did not."})
        analysis = await self.llm("B5", [last_step], messages, max_tokens=320)
        return {"test_path": test_path, "targeted": targeted,
                "full": full_result, "inspected_test_file": inspected_test,
                "decision_history": decision_history, "decision_status": status,
                "test_evidence_analysis": test_evidence_analysis,
                "test_analysis": analysis}

    async def run(self):
        await self.identify()
        plan = await self.llm("P0", [], [
            {"role": "system", "content": "You are a coding agent planner. Output only valid JSON."},
            {"role": "user", "content": self.question + "\nChoose concise search terms for "
                'source code and tests. Return {"source_term":"...","test_term":"..."}. ' 
                "Choose one specific single-word keyword per field, no spaces. "
                "Avoid the project name, generic words like 'test' or 'command', "
                "and made-up concatenations such as 'testcommand'. "
                "Prefer behavior-specific words from the task description."},
        ])
        choices = parse_json_reply(plan)
        source_term, test_term = str(choices["source_term"]), str(choices["test_term"])
        print(f"Plan: source={source_term!r}; tests={test_term!r}", flush=True)
        source, tests = await asyncio.gather(
            self.source_branch(source_term), self.test_branch(test_term))
        final = await self.llm("J0", ["A4", "B5"], [
            {"role": "system", "content":
             "You are a coding investigation agent. Respond with a single JSON object only. "
             "Be factual and don't claim you inspected files or ran tests not shown."},
            {"role": "user", "content": f"Question: {self.question}{self.profile_instructions}\n"
                f"Source investigation: {json.dumps(source)}\n"
                f"Testing investigation: {json.dumps(tests)}\n"
                'Give the PRIMARY implementation symbol (e.g. Class.method or function), '
                'implementation path, and best test path with one short explanation. '
                'Return JSON with EXACT keys: '
                '{"source_file":"src/click/...", "symbol":"Class.method or function", '
                '"test_file":"tests/...", "explanation":"..."}. '
                "Base all claims on observed results. No file edits."},
        ], max_tokens=340)
        return {"source": source, "tests": tests, "final_answer": final}


# Labels are evaluator-only: they are never included in model prompts.
# The test paths and implementation symbols have been checked against the
# public Pallets Click repository; local preflight checks guard version drift.
CASES = [
    {
        "id": "cli_runner",
        "question": "Investigate how Click's command-line testing helper runs a command and captures its output, return value, and exception. Locate the implementation and the related tests. Do not edit files.",
        "expected_source": "src/click/testing.py",
        "expected_symbol": "CliRunner.invoke",
        "expected_test": "tests/test_testing.py",
    },
    {
        "id": "confirmation",
        "question": "Investigate how Click displays a yes/no confirmation question to terminal users and interprets their response. Locate the implementation and the related tests. Do not edit files.",
        "expected_source": "src/click/termui.py",
        "expected_symbol": "confirm",
        "expected_test": "tests/test_termui.py",
    },
    {
        "id": "option_decorator",
        "question": "Investigate where Click implements the decorator that declares command-line options and how it attaches those options to commands. Locate the implementation and related tests. Do not edit files.",
        "expected_source": "src/click/decorators.py",
        "expected_symbol": "option",
        "expected_test": "tests/test_options.py",
    },
    {
        "id": "group_resolution",
        "question": "Investigate how a Click command group resolves a subcommand name into the command it will invoke. Locate the implementation and related tests. Do not edit files.",
        "expected_source": "src/click/core.py",
        "expected_symbol": "Group.resolve_command",
        "expected_test": "tests/test_commands.py",
    },
    {
        "id": "usage_formatter",
        "question": "Investigate how Click constructs and writes the usage line displayed in command help text. Locate the implementation and related tests. Do not edit files.",
        "expected_source": "src/click/formatting.py",
        "expected_symbol": "HelpFormatter.write_usage",
        "expected_test": "tests/test_formatting.py",
    },
]



def load_cases_file(path: Path) -> tuple[list[dict], str | None]:
    """Load evaluator-only Click cases. Never pass ground-truth labels to LLMs.

    The accepted format is {"repo_commit": "...", "cases": [...]} (or a list).
    Only the per-case `question` is sent to CodingPilot. The other fields are
    used exclusively for local preflight and post-run grading.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load cases file {path}: {exc}") from exc
    if isinstance(document, dict):
        cases = document.get("cases")
        expected_commit = document.get("repo_commit")
    else:
        cases, expected_commit = document, None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases JSON must contain a nonempty 'cases' list")
    if expected_commit is not None and (not isinstance(expected_commit, str) or
                                        not re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit)):
        raise ValueError("repo_commit must be a 40-character Git SHA")

    required = ("id", "question", "expected_source", "expected_symbol", "expected_test")
    seen_ids = set()
    selected = []
    for i, item in enumerate(cases, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{i} must be a JSON object")
        for key in required:
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Case #{i} has empty or missing {key!r}")
        case_id = item["id"]
        if not re.fullmatch(r"[a-z0-9_]+", case_id):
            raise ValueError(f"Invalid case id {case_id!r}: use lowercase letters, digits, _")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen_ids.add(case_id)
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", item["expected_symbol"]):
            raise ValueError(f"Invalid expected symbol in {case_id!r}")
        for field, prefix in (("expected_source", "src/click/"),
                              ("expected_test", "tests/")):
            rel = item[field]
            parts = Path(rel).parts
            if not rel.startswith(prefix) or not rel.endswith(".py") or                     Path(rel).is_absolute() or ".." in parts or "\\" in rel:
                raise ValueError(f"Unsafe {field} in {case_id!r}: {rel!r}")
        profile = item.get("investigation_profile", "basic")
        if profile not in ("basic", "test_evidence", "call_trace", "deep"):
            raise ValueError(f"Invalid investigation_profile for {case_id}: {profile!r}")
        record = {key: item[key] for key in required}
        if "investigation_profile" in item:
            record["investigation_profile"] = profile
        selected.append(record)
    return selected, expected_commit


PROFILE_INSTRUCTIONS = {
    "basic": "",
    "test_evidence": (
        " Additionally, inspect actual test-source code, identify one concrete test "
        "function and explain what behavior its assertions check. A passing pytest "
        "result alone is insufficient evidence."
    ),
    "call_trace": (
        " Additionally, trace a directly called source helper of the primary "
        "implementation (including a helper from a relative import if found), "
        "read its implementation and explain their relationship. If unavailable, "
        "report that limitation."
    ),
    "deep": (
        " Additionally, trace and read one directly called implementation helper, "
        "and inspect test source to explain a specific relevant test's behavior. "
        "If a helper is not found or tests do not cover the feature, state that."
    ),
}

def normalized_symbol_equal(found: str, expected: str) -> bool:
    """Accept a fully qualified spelling but not a different function."""
    val = found.strip().strip("` ")
    return val == expected or val.endswith("." + expected)


def grade_case(case: dict, run_result: dict | None, run_status: str) -> dict:
    """Separate implementation localization from exact reference-test matching.

    A focused pytest pass means the selected test module passes; it does not
    establish test relevance or patch correctness. Do not grant an alternate
    test file equivalence merely because its tests passed. Evaluator labels
    never enter the agent prompts.
    """
    empty = {
        "correct": False, "implementation_correct": False,
        "source_correct": False, "symbol_correct": False,
        "inspected_symbol_correct": False, "test_correct": False,
        "focused_pytest_passed": False, "focused_pytest_exit_code": None,
        "format_correct": False,
        "required_evidence_steps_complete": False,
        "test_source_inspected": False,
        "related_helper_available": False,
        "related_helper_inspected": False,
    }
    if run_status != "completed" or not isinstance(run_result, dict):
        return {**empty, "reason": "agent_error"}
    try:
        answer = parse_json_reply(run_result.get("final_answer", ""))
    except (ValueError, TypeError) as exc:
        return {**empty, "reason": f"invalid_final_json: {exc}"}

    required = ("source_file", "symbol", "test_file", "explanation")
    format_ok = all(isinstance(answer.get(key), str) and answer[key].strip()
                    for key in required)
    source = run_result.get("source") or {}
    tests = run_result.get("tests") or {}
    source_path = source.get("source_path")
    test_path = tests.get("test_path")
    source_ok = (source_path == case["expected_source"] and
                 answer.get("source_file") == case["expected_source"])
    test_ok = (test_path == case["expected_test"] and
               answer.get("test_file") == case["expected_test"])
    symbol_ok = normalized_symbol_equal(str(answer.get("symbol", "")),
                                        case["expected_symbol"])
    inspected_ok = normalized_symbol_equal(str(source.get("source_symbol", "")),
                                            case["expected_symbol"])
    targeted = tests.get("targeted") or {}
    pytest_ok = targeted.get("exit_code") == 0 and not targeted.get("timed_out", False)
    implementation_ok = source_ok and symbol_ok and inspected_ok
    exact_ok = source_ok and symbol_ok and test_ok
    profile = case.get("investigation_profile", "basic")
    needs_test = profile in ("test_evidence", "deep")
    needs_helper = profile in ("call_trace", "deep")
    test_evidence_ok = bool(tests.get("inspected_test_file"))
    helper_available = bool(source.get("related_helper_available"))
    helper_inspected = bool(source.get("related_helper"))
    evidence_steps_complete = ((not needs_test or test_evidence_ok) and
                               (not needs_helper or not helper_available or helper_inspected))

    return {
        "correct": exact_ok,  # legacy metric: strict source + symbol + test reference
        "implementation_correct": implementation_ok,
        "source_correct": source_ok,
        "symbol_correct": symbol_ok,
        "inspected_symbol_correct": inspected_ok,
        "test_correct": test_ok,  # strict reference test-file match, NOT relevance
        "focused_pytest_passed": pytest_ok,
        "focused_pytest_exit_code": targeted.get("exit_code"),
        "format_correct": format_ok,
        "workload_profile": profile,
        "required_evidence_steps_complete": evidence_steps_complete,
        "test_source_inspected": test_evidence_ok,
        "related_helper_available": helper_available,
        "related_helper_inspected": helper_inspected,
        "reason": "exact_source_symbol_test_match" if exact_ok else "answer_mismatch",
        "observed": {
            "source_tool_path": source_path,
            "test_tool_path": test_path,
            "final": answer,
        },
        "note": (
            "implementation_correct requires the agent to select, inspect, and "
            "report the reference symbol. correct retains strict source/symbol/" 
            "test-file matching. A passing focused pytest alone does not "
            "establish test relevance or semantic task correctness."
        ),
    }


def preflight(case: dict, repo: Path):
    """Fail early if test labels do not exist in user's checked-out repo."""
    for key in ("expected_source", "expected_test"):
        p = repo / case[key]
        if not p.is_file():
            raise RuntimeError(f"{case['id']}: missing {p}; check Click version")
    source = (repo / case["expected_source"]).read_text(encoding="utf-8")
    method = case["expected_symbol"].split(".")[-1]
    if not re.search(r"^\s*(?:async\s+)?def\s+" + re.escape(method) + r"\s*\(", source, re.M):
        raise RuntimeError(f"{case['id']}: expected symbol {method} absent in local repo")


async def main():
    import subprocess
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="coding-repos/click")
    parser.add_argument("--api", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--max-inflight-llm", type=int, default=2)
    parser.add_argument("--test-timeout", type=int, default=90)
    parser.add_argument("--max-followups", type=int, default=1)
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Select case ids from the active dataset; default runs all")
    parser.add_argument("--cases-file", type=Path, default=None,
                        help="Optional external JSON dataset; defaults to the built-in five")
    parser.add_argument("--out", default="results/coding_agent_correctness_5.json")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / "src/click").is_dir() or not (repo / "tests").is_dir():
        parser.error(f"Not a Click repository: {repo}")
    if args.max_followups < 0 or args.max_inflight_llm < 1:
        parser.error("--max-followups must be >= 0 and --max-inflight-llm >= 1")
    if args.cases_file:
        try:
            active_cases, pinned_commit = load_cases_file(args.cases_file)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        active_cases, pinned_commit = CASES, None
    if args.cases is not None:
        if not args.cases:
            parser.error("--cases requires at least one case id")
        if len(set(args.cases)) != len(args.cases):
            parser.error("Duplicate case IDs in --cases")
        chosen = set(args.cases)
        unknown = chosen - {c["id"] for c in active_cases}
        if unknown:
            parser.error(f"Unknown case(s): {sorted(unknown)}")
        selected = [c for c in active_cases if c["id"] in chosen]
    else:
        selected = list(active_cases)
    for case in selected:
        preflight(case, repo)
    rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    git_commit = rev.stdout.strip() if rev.returncode == 0 else None
    if pinned_commit and (not git_commit or git_commit.lower() != pinned_commit.lower()):
        parser.error(f"Cases file pins Click commit {pinned_commit}, but repository is "
                     f"at {git_commit or 'an unknown commit'}. Use the pinned checkout.")
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True)
    repo_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    all_rows = []
    print(f"Running {len(selected)} Click investigation cases. "
          f"Repo commit={git_commit} dirty={repo_dirty}", flush=True)
    for i, case in enumerate(selected, 1):
        print(f"\n=== Case {i}/{len(selected)}: {case['id']} ===", flush=True)
        pilot = CodingPilot(repo, args.api, sys.executable, args.max_inflight_llm,
                            args.test_timeout, args.max_followups)
        pilot.investigation_profile = case.get("investigation_profile", "basic")
        pilot.question = case["question"]
        pilot.profile_instructions = PROFILE_INSTRUCTIONS[pilot.investigation_profile]
        print(f"Investigation profile: {pilot.investigation_profile}", flush=True)
        case_start_wall_epoch_s = time.time()
        status, result, err = "completed", None, None
        try:
            result = await pilot.run()
        except Exception as exc:
            status, err = "failed", repr(exc)
            print(f"Case failed: {err}", file=sys.stderr, flush=True)
        case_end_wall_epoch_s = time.time()
        grade = grade_case(case, result, status)
        print("Grade: " + json.dumps({
            "source": grade["source_correct"],
            "implementation_correct": grade["implementation_correct"],
            "symbol": grade["symbol_correct"],
            "inspected_symbol": grade.get("inspected_symbol_correct"),
            "test": grade["test_correct"],
            "focused_pytest_passed": grade.get("focused_pytest_passed"),
            "localization_correct": grade["correct"]}), flush=True)
        all_rows.append({
            "case_id": case["id"], "question": case["question"],
            "investigation_profile": pilot.investigation_profile,
            "expected": {"source": case["expected_source"],
                         "symbol": case["expected_symbol"],
                         "test": case["expected_test"]},
            "status": status, "error": err,
            "case_start_wall_epoch_s": case_start_wall_epoch_s,
            "case_end_wall_epoch_s": case_end_wall_epoch_s,
            "model": pilot.model,
            "events": pilot.events, "result": result, "grade": grade,
            "duration_s": pilot.now(),
        })
    summary = {
        "attempted": len(all_rows),
        "agent_executed": sum(r["status"] == "completed" for r in all_rows),
        "exact_localization_correct": sum(r["grade"]["correct"] for r in all_rows),
        "implementation_localization_correct": sum(
            r["grade"]["implementation_correct"] for r in all_rows),
        "focused_pytest_passed": sum(
            r["grade"].get("focused_pytest_passed", False) for r in all_rows),
        "exact_localization_and_test_passed": sum(
            r["grade"]["correct"] and r["grade"].get("focused_pytest_passed") is True
            for r in all_rows),
        "source_correct": sum(r["grade"]["source_correct"] for r in all_rows),
        "symbol_correct": sum(r["grade"]["symbol_correct"] for r in all_rows),
        "inspected_symbol_correct": sum(r["grade"].get("inspected_symbol_correct", False) for r in all_rows),
        "test_correct": sum(r["grade"]["test_correct"] for r in all_rows),
        "format_correct": sum(r["grade"]["format_correct"] for r in all_rows),
        "required_evidence_steps_complete": sum(
            r["grade"].get("required_evidence_steps_complete", False) for r in all_rows),
    }
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "repo": str(repo), "git_commit": git_commit, "repo_dirty": repo_dirty,
               "config": {"api": args.api, "max_inflight_llm": args.max_inflight_llm,
                          "max_followups": args.max_followups,
                          "test_timeout_seconds": args.test_timeout,
                          "artificial_tool_delay_seconds": 0},
               "evaluation": "code_investigation_feasibility_grading_v7",
               "dataset": {"source": str(args.cases_file) if args.cases_file else "built_in_5",
                           "total_cases": len(active_cases),
                           "selected_cases": len(selected),
                           "pinned_commit": pinned_commit},
               "summary": summary, "cases": all_rows}
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSUMMARY {json.dumps(summary)}")
    print(f"Saved results: {dest}")


if __name__ == "__main__":
    asyncio.run(main())
