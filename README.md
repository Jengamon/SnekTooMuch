SnekTooMuch - A Snek Interpreter That's a Bit Much
==================================================

## Overview

I put a little too much time into this, so it ended up a bit large.
It interprets the 6.009 variant of LISP, Snek, but with a few additions:

- Implicit multibodies (added `begin`) for `define`, `let`, `letrec` and `lambda`
- `letrec`, a version of `let` that defines the environment and its added names before evaluating the bound values
- `quote`, `unquote` for basic quoting and unquoting and `quasiquote` environment with support for `unquote` and `unquote-splicing` for generation tomfoolery
- Token equivalents for {`quote` => `'`, `quasiquote` => `` ` ``, `unquote` => `,`, `unquote-splicing` => `,@`}
- Non-standard builtins: {`int`, `set-car`, `py-import`, `getattr`, `num?`, `list?`, `display`, `join`}
    - `int` (1 args) -> returns the argument passed through python's `int` builtin
    - `set-car` (2 args: cons val) -> mutates the given cons `car` value to val
    - `py-import` (1 arg) -> returns a python module with the given name (use `quote` to give a name)
    - `getattr` (2 args: module name) -> returns `getattr(module, name)` (use `quote` to give a name)
    - `num?` (1 arg) -> returns if argument is numerical (int or float)
    - `list?` (1 arg) -> returns if argument is list (list cons or nil) [iterative check, so a given cons is checked to make sure it is a *list* one]
    - `display` (variadric) -> Prints the result of joining all it's arguments (converted with python's `str`) with spaces, and returns that string
    - `join` (1+ variadric) -> joins its arguments using the first given argument as a separator
- Rest parameters: following a `.` with a single other name will allow lambdas to handle more arguments than their signature suggests, as the runtime
    will pass the rest of the arguments into the lambda as a list using this name
- The `(define (x) 3)` define lambda syntax is now available for `let`/`letrec` bindings: `(let (((x) 3)) x)` will return the same thing, but doesn't define name `x` in the surrounding environment
- New variadric `lambda` syntax (from Scheme, I think): `(lambda args args)` will return a list of all arguments given to it.
- REPL colors if you run `pip install termcolor colorama` before opening the repl
- Support for commandline flags
    - `-p` or `--no-color` for plain output (w/ no colors)
    - `-s` or `--no-multiexp` for no multiexpressions
    - `-d` or `--no-rest` to disable rest parameters

## Things that could probably be added

- A proper string implementation, I just use `quote` whenever I want a string, but that doesn't allow for strings 
with spaces. (If added, check for interactions with the fact that the representation of `'ident` for any ident is in fact a Python string)
- Umm, it is possible to generate lists that are uninterpretable due to Python recursion limits. See if that's fixable.