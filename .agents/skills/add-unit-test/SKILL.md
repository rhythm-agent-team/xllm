---
name: add-unit-test
description: Add or update xLLM C++/Python unit tests. Use when creating tests, wiring test targets, or checking naming, dependencies, and platform gates.
---

# Add Unit Test

## Python tests

Read **Python tests** in `xllm/python/README.md`. Put Python tests in
`tests/python/`. Use normal imports; `conftest.py` loads `xllm_export` and calls
`initialize_runtime()` before collection. Do not repeat initialization or add
test-local extension loading or `sys.path` manipulation.

Run `python -m pytest tests/python/<file>.py` from the checkout root against matching
native build artifacts in the environment required by checkout instructions.
Direct pytest does not build the extension.

## C++ test workflow

1. Inspect the production code and the nearest existing tests before writing a new test.
   - Match the production path under `xllm/` to `tests/` where possible.
   - Prefer extending an existing nearby `*_test.cpp` and `cc_test` target when the behavior belongs to the same domain.
   - Create a new test source only when it improves isolation, keeps platform setup separate, or follows an existing directory pattern.

2. Read the project style guide before editing production files under `xllm/`, and apply the same C++ style discipline to new test code:
   `.agents/skills/code-review/references/custom-code-style.md`.

3. Follow the current test layout and CMake conventions.
   - Read [xllm-test-patterns.md](references/xllm-test-patterns.md) when adding a new test file, new `cc_test`, platform-specific test, or test directory.
   - Use `*_test.cpp` for C++ test files and `*_test.cu` for CUDA source tests.
   - Do not create nested `test/` or `tests/` directories for new unit tests unless the surrounding tree already requires that structure.

4. Wire tests through CMake with `include(cc_test)` and `cc_test(...)`.
   - Keep source names relative to the current test directory unless an existing target already uses an absolute source path for a production `.cpp`.
   - Use target names ending in `_test`.
   - Put platform-directory gates in the parent `CMakeLists.txt` when the whole child directory is platform-specific.
   - Use target-level `if(USE_NPU)`, `if(USE_MLU)`, `if(USE_CUDA)`, or generator expressions only when a mixed directory contains both generic and platform-specific tests.

5. Write tests for observable behavior, not implementation trivia.
   - Cover success, edge, and error paths touched by the change.
   - Prefer deterministic inputs, fixed seeds, and small tensors/data structures.
   - Keep helpers file-local in an anonymous namespace unless shared by multiple test files.
   - Use `TEST`/`TEST_F` names that describe behavior clearly.

6. Validate narrowly before finishing.
   - Always run `git diff --check` for the changed test paths.
   - Search for stale filenames after moving or renaming tests.
   - Run the narrowest relevant build or test command in the required execution environment; if blocked, report the reason and checks performed.

## Common Commands

```bash
rg --files tests/<area>
rg "old_test_name|old_file_name" tests xllm CMakeLists.txt
git diff --check -- tests/<area>
```

Follow checkout instructions for the execution environment and build/test commands.
