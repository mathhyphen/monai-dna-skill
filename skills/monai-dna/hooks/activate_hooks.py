"""
activate_hooks.py - Programmatic hook registration for MONAI DNA guardrails.

This module provides functions to load and register the MONAI DNA PreToolUse hooks.
It is used by the monai-dna skill system to activate session-duration hooks when
the skill is invoked.

Usage:
    import activate_hooks
    hooks = activate_hooks.load_guardrails()
    activate_hooks.register(hooks)

Note: This module is provided for flexibility. The primary activation path is
through the skill definition in SKILL.md, which references guardrails.json.
This module is useful for:
    - Testing hooks in isolation
    - Custom skill loading mechanisms
    - Standalone validation of hook definitions
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

HOOKS_FILE = Path(__file__).parent / "guardrails.json"


def load_guardrails() -> list[dict[str, Any]]:
    """Load guardrail hook definitions from guardrails.json.

    Returns:
        A list of hook definition dictionaries following the Claude Code
        PreToolUse hook schema.

    Raises:
        FileNotFoundError: If guardrails.json does not exist.
        json.JSONDecodeError: If guardrails.json contains invalid JSON.
    """
    with open(HOOKS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pre_tool_use", [])


def validate_hook(hook: dict[str, Any]) -> list[str]:
    """Validate a single hook definition and return a list of error messages.

    A hook is valid if it has:
        - name: non-empty string
        - description: non-empty string
        - matcher: dict with tool_name and params_match
        - action: "block" or "warn"
        - message: non-empty string

    Args:
        hook: A hook definition dictionary.

    Returns:
        A list of error strings (empty if the hook is valid).
    """
    errors: list[str] = []
    required_fields = ["name", "description", "matcher", "action", "message"]
    for field in required_fields:
        if field not in hook:
            errors.append(f"Missing required field: '{field}'")

    if "action" in hook and hook["action"] not in ("block", "warn"):
        errors.append(f"action must be 'block' or 'warn', got: '{hook['action']}'")

    if "matcher" in hook:
        matcher = hook["matcher"]
        if "tool_name" not in matcher:
            errors.append("matcher missing 'tool_name'")
        if "params_match" not in matcher:
            errors.append("matcher missing 'params_match'")
        elif not isinstance(matcher["params_match"], dict):
            errors.append("matcher.params_match must be a dict")

        # Validate regex patterns
        if "params_match" in matcher:
            for param, pattern in matcher["params_match"].items():
                try:
                    re.compile(pattern)
                except re.error as e:
                    errors.append(f"Invalid regex in {param}: {e}")

    return errors


def validate_all() -> dict[str, list[str]]:
    """Validate all hooks in guardrails.json.

    Returns:
        A dict mapping hook names to lists of error messages.
        Hooks with no errors are omitted from the dict.
    """
    hooks = load_guardrails()
    result: dict[str, list[str]] = {}
    for hook in hooks:
        name = hook.get("name", "<unnamed>")
        errors = validate_hook(hook)
        if errors:
            result[name] = errors
    return result


def compile_matchers() -> list[tuple[str, str, re.Pattern]]:
    """Compile all regex matchers for fast runtime evaluation.

    Returns:
        A list of (tool_name, param_name, compiled_regex) tuples.
    """
    hooks = load_guardrails()
    matchers: list[tuple[str, str, re.Pattern]] = []
    for hook in hooks:
        matcher = hook.get("matcher", {})
        tool_name = matcher.get("tool_name", "")
        params_match = matcher.get("params_match", {})
        for param, pattern in params_match.items():
            matchers.append((tool_name, param, re.compile(pattern)))
    return matchers


def register(hooks: list[dict[str, Any]] | None = None) -> None:
    """Register hooks with the active Claude Code session.

    This function is called by the skill activation system. It takes
    the loaded hook definitions and registers them as PreToolUse hooks.

    Args:
        hooks: Optional list of hook definitions. If None, loads from guardrails.json.

    Note:
        In the Claude Code skill system, hooks are registered through the
        skill definition file (SKILL.md) referencing guardrails.json.
        This function is provided for programmatic use cases such as testing
        or custom skill loaders.
    """
    if hooks is None:
        hooks = load_guardrails()

    validation_errors = validate_all()
    if validation_errors:
        error_lines = []
        for name, errors in validation_errors.items():
            error_lines.append(f"  {name}:")
            for err in errors:
                error_lines.append(f"    - {err}")
        raise ValueError(
            f"H Hook definition errors found:\n" + "\n".join(error_lines)
        )

    # In the Claude Code skill system, registration is handled by the agent
    # framework when the skill is activated. This function validates and
    # prepares the hooks for that registration.
    #
    # The actual registration call depends on the Claude Code SDK:
    #   from claude_code_sdk import ClaudeCode
    #   client = ClaudeCode()
    #   client.set_pre_tool_use_hooks(hooks)
    #
    # For standalone validation, call validate_all() instead.
    return


if __name__ == "__main__":
    import sys

    errors = validate_all()
    if errors:
        print("Hook validation errors:", file=sys.stderr)
        for name, errs in errors.items():
            print(f"  {name}:", file=sys.stderr)
            for e in errs:
                print(f"    - {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"All {len(load_guardrails())} hooks are valid.")
