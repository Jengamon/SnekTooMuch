#!/usr/bin/env python3
"""6.009 Lab 9: Snek Interpreter"""

import doctest
import sys
TURTLE_ENABLED = False
try:
    if TURTLE_ENABLED:
        from cturtle import turtle
    else:
        turtle = None
except Exception:
    turtle = None
# NO ADDITIONAL IMPORTS!


###########################
# Snek-related Exceptions #
###########################

class SnekError(Exception):
    """
    A type of exception to be raised if there is an error with a Snek
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """
    pass


class SnekSyntaxError(SnekError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """
    pass


class SnekNameError(SnekError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """
    pass


class SnekEvaluationError(SnekError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SnekNameError.
    """
    pass


############################
# Tokenization and Parsing #
############################

def is_num(x):
    """Checks if x is a digit"""
    return ord('0') <= ord(x) <= ord('9')

def is_alpha(x):
    """Checks if x is [A-Za-z]"""
    return ord('A') <= ord(x) <= ord('z')

def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Snek
                      expression
    """
    def _tokenize(source):
        """Generates tokens from a source string."""
        idx = 0
        terminal = lambda c: c in [' ', '\t', '\n', '\r', '(', ')']
        def skip_whitespace():
            nonlocal idx
            """Advances the index until it points to a non-whitespace character, and returns it"""
            c = lambda: source[idx] if idx < len(source) else ' '
            while idx < len(source) and (c() == ' ' or c() == '\t' or c() == '\n' or c() == '\r'):
                idx += 1
            return c()
        def parse_symbol():
            """Parse a valid identifier, such as 'x' or 'boogey_man' or 'c2'"""
            nonlocal idx
            ident = ""
            while idx < len(source) and not terminal(source[idx]):
                ident += source[idx]
                idx += 1
            idx -= 1 # Backtrack, bc last character is *invalid* and loop assumes we stop on a valid token character
            return ident
        def parse_number():
            """Parses valid numbers like '3' or '3.42'"""
            nonlocal idx
            num = ""
            def parse_digits():
                nonlocal idx
                num = ""
                while idx < len(source) and is_num(source[idx]):
                    num += source[idx]
                    idx += 1
                return num
            # Parse initial numbers
            oidx = idx
            num += parse_digits()
            if idx < len(source) and source[idx] == '.': # if we find a dot
                # Parse out the second part of the number string
                idx += 1
                num += ("." + parse_digits())
            if idx < len(source) and not terminal(source[idx]): # the number didn't terminate... this is an identifier
                idx = oidx
                return parse_symbol()
            idx -= 1 # Backtrack, bc last character is *invalid* and loop assumes we stop on a valid token character
            return num
        while idx < len(source):
            c = skip_whitespace()
            if c == '(':
                yield '('
            elif c == ')':
                yield ')'
            elif c == ';': # Handle comments, by skip to the newline, then the loop consumes the newline
                while idx < len(source) and source[idx] != '\n':
                    idx += 1
            elif is_num(c):
                yield parse_number()
            elif c == '-':
                idx += 1
                if is_num(source[idx]):
                    yield '-' + parse_number()
                else:
                    idx -= 1
                    yield '-'
            elif c == '#':
                idx += 1
                if source[idx] == 't' or source[idx] == 'T':
                    yield '#t'
                elif source[idx] == 'f' or source[idx] == 'F':
                    yield '#f'
                else:
                    idx -= 1
                    yield '#'
            else:
                yield parse_symbol()
            idx += 1
    return list(filter(lambda x: not not x, _tokenize(source)))


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    # Check first token, it can be one of 3 things if it is valid:
    # - '(' for S-expressions
    # - a string beginning with a digit for numbers
    # - a string beginning with a non-digit for symbols
    def parse_item(tokens):
        if tokens:
            token = tokens.pop(0)
            if token == '(':
                # Parse out S-expression
                expr = []
                while len(tokens) > 0 and tokens[0] != ')':
                    item = parse_item(tokens)
                    expr.append(item)
                if len(tokens) > 0 and tokens[0] == ')':
                    tokens.pop(0)
                else:
                    raise SnekSyntaxError

                # Set form checker
                def check_set_form_length(leng):
                    if len(expr) != leng:
                        raise SnekSyntaxError
                if expr: # If expr is not empty
                    if expr[0] == 'define':
                        check_set_form_length(3)
                        if not (isinstance(expr[1], str) or (isinstance(expr[1], list) and len(expr[1]) > 0 and (all(map(lambda i: isinstance(i, str), expr[1]))))):
                            raise SnekSyntaxError
                    elif expr[0] == 'lambda':
                        check_set_form_length(3)
                        if not (isinstance(expr[1], list) and all(map(lambda i: isinstance(i, str), expr[1]))):
                            raise SnekSyntaxError
                    elif expr[0] == 'if':
                        check_set_form_length(4)
                    elif expr[0] == 'let':
                        check_set_form_length(3)
                        if not (isinstance(expr[1], list) and all(map(lambda i: isinstance(i, list) and len(i) == 2 and isinstance(i[0], str), expr[1]))):
                            raise SnekSyntaxError
                    elif expr[0] == 'set!':
                        check_set_form_length(3)
                        if not isinstance(expr[1], str):
                            raise SnekSyntaxError("set! must be a symbol followed by a value")
                return expr
            elif is_num(token[0]) or (len(token) > 1 and token[0] == '-' and is_num(token[1])):
                try:
                    # Parse a number from the token
                    if sum(map(lambda c: 1 if c == '.' else 0, token)) == 1: # We have exactly 1 decimal point, a valid float
                        return float(token)
                    elif sum(map(lambda c: 1 if c == '.' else 0, token)) == 0: # We have 0 decimal points, a valid int
                        return int(token)
                    else: # Some weird numerical identifier
                        return token
                except ValueError: # Something went wrong in parsing, treat as identifier
                    return token
            elif token == '#t':
                return True
            elif token == '#f':
                return False
            elif token == 'nil':
                return Nil()
            elif token == ')': # Expected a expression, got an expression ender
                raise SnekSyntaxError
            else:
                return token
        else:
            raise SnekSyntaxError # Expected somthing, got nothing
    top_level = parse_item(tokens)
    if len(tokens) != 0:
        raise SnekSyntaxError # We expected all tokens consumed, but this didn't happen, so something went wrong
    return top_level


######################
# Built-in Functions #
######################

def product(args):
    v = 1
    for i in args:
        v *= i
    return v

def division(args):
    x = args[0]
    for i in args[1:]:
        x = (x / i)
    return x

def all_equal(args): # =?
    if args:
        item = args[0]
        for o in args[1:]:
            if o != item:
                return False
        return True
    else:
        return True

def comparison(op):
    """
    A generic comparison function generator, that produces a function that implements a comparsion using the lambda op between the current and other values.
    If op is truthy, continue the comparison,
    If op is falsy, return False.
    If all items pass op, return True.
    """
    def comp(args):
        if args:
            item = args[0]
            for o in args[1:]:
                if op(item, o):
                    item = o
                else:
                    return False
            return True
        else:
            return True
    return comp

def not_fn(args):
    if len(args) != 1:
        raise SnekEvaluationError("not can only take 1 argument")
    return not args[0]

def cons(args):
    if len(args) != 2:
        raise SnekEvaluationError("cons can only take 2 arguments")
    return Pair(args[0], args[1])

def car(args):
    if len(args) != 1 or not isinstance(args[0], Pair):
        raise SnekEvaluationError("car can only take 1 cons argument")
    item = args[0]
    return item.car

def cdr(args):
    if len(args) != 1 or not isinstance(args[0], Pair):
        raise SnekEvaluationError("cdr can only take 1 cons argument")
    item = args[0]
    return item.cdr

def list_snek(args):
    if not args:
        return Nil()
    else:
        return Pair(args[0], list_snek(args[1:]))

def length(args):
    if len(args) != 1 or not (isinstance(args[0], Pair) or args[0] == Nil()):
        raise SnekEvaluationError("length can only take 1 list cons argument")
    i = 0
    item = args[0]
    while item != Nil():
        i += 1
        item = item.cdr
        if not (isinstance(item, Pair) or item == Nil()):
            raise SnekEvaluationError("Only supports list cons, not arbitrary cons")
    return i

def elt_at_index(args):
    if len(args) != 2 or not (isinstance(args[0], Pair) and isinstance(args[1], int)) or args[1] < 0:
        raise SnekEvaluationError("elt-at-index can only take 2 arguments: a list cons and positive integer")
    if args[1] == 0:
        return args[0].car
    else:
        return elt_at_index([args[0].cdr, args[1] - 1])

def concat(args):
    if not args:
        return Nil()
    else:
        if not (isinstance(args[0], Pair) or args[0] == Nil()):
            raise SnekEvaluationError
        olst = args[0].clone()
        lst = olst
        other = concat(args[1:])
        # Find the Nil valued cdr, and set that Pair's cdr to the "other" Pair
        # Special case: if lst is nil, just return the other
        if lst == Nil():
            return other
        else:
            while lst.cdr != Nil():
                if isinstance(lst.cdr, Pair):
                    lst = lst.cdr
                else:
                    raise SnekEvaluationError("Only supports list cons, not arbitrary cons")
            lst.cdr = other
        return olst

def map_snek(args):
    if len(args) != 2 or not (isinstance(args[1], Pair) or args[1] == Nil()):
        raise SnekEvaluationError
    new_list = args[1].clone()
    if args[1] != Nil():
        try:
            new_list.car = args[0]([new_list.car])
        except TypeError:
            raise SnekEvaluationError("Could not call arg 0 as function")
        if new_list.cdr != Nil():
            new_list.cdr = map_snek([args[0], new_list.cdr])
    return new_list

def filter_snek(args):
    if len(args) != 2 or not (isinstance(args[1], Pair) or args[1] == Nil()):
        raise SnekEvaluationError
    new_list = args[1].clone()
    if args[1] != Nil():
        try:
            cond = args[0]([new_list.car])
            if not cond:
                new_list = filter_snek([args[0], new_list.cdr])
        except TypeError:
            raise SnekEvaluationError("Could not call arg 0 as function")
        if new_list != Nil() and new_list.cdr != Nil():
            new_list.cdr = filter_snek([args[0], new_list.cdr])
    return new_list

def reduce_snek(args):
    if len(args) != 3 or not (isinstance(args[1], Pair) or args[1] == Nil()):
        raise SnekEvaluationError
    val = args[2]
    cons = args[1]
    while cons != Nil():
        try:
            val = args[0]([val, cons.car])
            cons = cons.cdr
            if not (isinstance(cons, Pair) or cons == Nil()):
                raise SnekEvaluationError("Only supports list cons, not arbitrary cons")
        except TypeError:
            raise SnekEvaluationError("Could not call arg 0 as function")
    return val

def begin_snek(args):
    return args[-1]

def int_snek(args):
    if len(args) != 1:
        raise SnekEvaluationError("int can only take 1 argument")
    return int(args[0])

snek_builtins = {
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    '*': product,
    '/': division,
    '=?': all_equal,
    '>': comparison(lambda i, o: i > o),
    '>=': comparison(lambda i, o: i >= o),
    '<': comparison(lambda i, o: i < o),
    '<=': comparison(lambda i, o: i <= o),
    'not': not_fn,
    'cons': cons,
    'car': car,
    'cdr': cdr,
    'list': list_snek,
    'length': length,
    'elt-at-index': elt_at_index,
    'concat': concat,
    'map': map_snek,
    'filter': filter_snek,
    'reduce': reduce_snek,
    'begin': begin_snek,
    'int': int_snek,
}


##############
# Evaluation #
##############

class Environment:
    """Defines an environment for Snek code"""
    def __init__(self, parent=None, locked=False):
        """Intializes an environment with an optional parent environment."""
        self.parent = parent
        self.bindings = {}
        self.locked = locked

    def __repr__(self):
        return "{} [parent={}]".format(self.bindings, self.parent)

    def define(self, label, obj):
        """Defines a label as a certain object in this environment."""
        self.bindings[label] = obj

    def lookup(self, label):
        """
        Searches this environment, and its parents if necessary for a label, and returns the object if it exists.
        Raises SnekNameError on failure.
        """
        if label in self.bindings:
            return self.bindings[label]
        else:
            if self.parent:
                return self.parent.lookup(label)
            else:
                raise SnekNameError("name '{}' is not defined".format(label))

    def delete(self, label):
        """Removes a name and its value from an environment (or that of its parents). (will not delete from locked environments.)"""
        if label in self.bindings:
            if not self.locked:
                del self.bindings[label]
        else:
            if self.parent:
                self.parent.delete(label)


    def defined_names(self, tree=False):
        """Returns the defined names of this scope, and of all its parents if tree = True"""
        if not tree:
            return list(self.bindings.keys())
        else:
            return list(self.bindings.keys()) + (list(self.parent.defined_names(tree=True)) if self.parent else [])

class UserFunction:
    """Represents functions defined in Snek code"""
    def __init__(self, parent, params, body):
        """
        Intializer.
        parent is the encosing encironment from creation time, and
        params are a list of names to bind arguments to when called.
        body in the code to execute when this function is called
        """
        self.parent = parent
        self.params = params
        self.body = body

    def __call__(self, args):
        """
        Call this function.
        Creates a new environment, and expects *args to be evaluated values.
        """

        func_env = Environment(self.parent)
        self.define_args(args, func_env)
        return evaluate(self.body, func_env)

    def define_args(self, args, env):
        """Define the params using the args given in a given environment"""
        if len(self.params) != len(args):
            raise SnekEvaluationError("wrong number of arguments (expected {}, got {})".format(len(self.params), len(args)))
        for (i, param) in enumerate(self.params):
            env.define(param, args[i])

    def __repr__(self):
        pl = len(self.params)
        return "function object ({} arg{})".format(pl, "s" if pl != 1 else "")

class Pair:
    """A LISP 'non atomic S-expression'"""
    def __init__(self, car=None, cdr=None):
        self.car = car
        self.cdr = cdr

    def __repr__(self):
        return "[{} {}]".format(self.car, self.cdr)

    def clone(self):
        """Creates an independant copy of this Pair"""
        return Pair(self.car, self.cdr)

class Nil:
    """Used to represent the Nil object."""
    def __eq__(self, o):
        if isinstance(o, Nil):
            return True
        else:
            return False

    def __repr__(self):
        return "nil"

    def clone(self):
        """For convenience. Returns self, due to how Nil works."""
        return self

builtin_env = Environment(locked=True)

for name in snek_builtins:
    builtin_env.define(name, snek_builtins[name])

def evaluate(tree, env=None):
    """
    Evaluate the given syntax tree according to the rules of the Snek
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if not env:
        env = Environment(builtin_env)
    restore = {} # Hold onto values we butcher for tail calls, then restore after we are done looping
    # print(tree)
    def execute():
        """Main evaluator loop."""
        nonlocal env, tree, restore
        loopc = 0 # Loop counter
        loopt = 1 # Loop target (how many times we want to loop)
        def tailcall(code, _env=None):
            """Tail-calls the given code, by replacing the tree and then increasing the loop target"""
            nonlocal loopt, tree, env
            if _env:
                env = _env
            tree = code
            loopt += 1
        while loopc < loopt:
            if isinstance(tree, str):
                return env.lookup(tree)
            elif isinstance(tree, int) or isinstance(tree, float) or isinstance(tree, bool) or isinstance(tree, Nil):
                return tree  
            elif isinstance(tree, list): # S-expression
                if len(tree) < 1:
                    raise SnekEvaluationError
                if tree[0] == 'define': # Definition time
                    name = None
                    value = None
                    if isinstance(tree[1], str):
                        name = tree[1]
                        value = evaluate(tree[2], env)
                    elif isinstance(tree[1], list):
                        name = tree[1][0]
                        params = tree[1][1:]
                        body = tree[2]
                        value = UserFunction(env, params, body)
                    env.define(name, value)
                    return value
                elif tree[0] == 'lambda': # Lambda time
                    params = tree[1]
                    body = tree[2]
                    return UserFunction(env, params, body)
                elif tree[0] == 'if':
                    cond = evaluate(tree[1], env)
                    if cond:
                        tailcall(tree[2])
                        #return evaluate(tree[2], env)
                    else:
                        tailcall(tree[3])
                        #return evaluate(tree[3], env)
                elif tree[0] == 'and':
                    x = True
                    for expr in tree[1:]:
                        x = x and evaluate(expr, env)
                        if not x:
                            break
                    return x
                elif tree[0] == 'or':
                    x = False
                    for expr in tree[1:]:
                        x = x or evaluate(expr, env)
                        if x:
                            break
                    return x
                elif tree[0] == 'let':
                    bindings = []
                    for pair in tree[1]:
                        bindings.append((pair[0], evaluate(pair[1], env)))
                    let_env = Environment(env)
                    for (name, value) in bindings:
                        let_env.define(name, value)
                    tailcall(tree[2], let_env)
                    # return evaluate(tree[2], let_env)
                elif tree[0] == 'set!':
                    target_env = env
                    while True:
                        if tree[1] in target_env.defined_names():
                            val = evaluate(tree[2], env)
                            target_env.define(tree[1], val)
                            return val
                        else:
                            if target_env.parent:
                                target_env = target_env.parent
                            else: # We've walked all parent, and haven't found the name. Fail.
                                raise SnekNameError("set! targeting not existant binding")
                elif tree[0] == 'turtle':
                    name = tree[1]
                    args = list(map(lambda exp: evaluate(exp, env), tree[2:]))
                    if turtle:
                        return turtle(name, args)
                else: # Environment call
                    target = evaluate(tree[0], env)
                    args = list(map(lambda subn: evaluate(subn, env), tree[1:]))
                    try:
                        if type(target) is UserFunction:
                            # Do some tail-calling
                            # Make sure to keep the environments clean
                            env_names = env.defined_names()
                            for name in target.params:
                                if name not in restore:
                                    if name in env_names:
                                        restore[name] = env.lookup(name)
                                    else:
                                        restore[name] = None
                            target.define_args(args, env)
                            tailcall(target.body)
                        else:
                            return target(args)
                    except TypeError as e: # Cannot call target as a function
                        raise SnekEvaluationError(e)
            else:
                raise SnekEvaluationError # Unexpected type encountered
            loopc += 1
    ret = execute()
    # Restore the stuff we butchered
    for name in restore:
        val = restore[name]
        if val == None:
            env.delete(name)
        else:
            env.define(name, val)
    return ret

def result_and_env(tree, env=None):
    """Debugging function. Returns the result, along with the environment it was executed in"""
    if not env:
        env = Environment(builtin_env)
    result = evaluate(tree, env)
    return (result, env)

def evaluate_file(filename, env=None):
    """Evaluates filename from working directory, assuming 1 expression in file, and returns that expression evaluated"""
    if not env:
        env = Environment(builtin_env)
    with open(filename, 'r') as f:
        source = "\n".join(f.readlines())
    tokens = tokenize(source)
    tree = parse(tokens)
    return evaluate(tree, env)

if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()

    repl_env = Environment(builtin_env)

    for arg in sys.argv[1:]:
        evaluate_file(arg, repl_env)

    while True:
        string = input("in> ")
        if string.lower() == "quit":
            break
        else:
            try:
                tokens = tokenize(string)
                tree = parse(tokens)
                print("  out>", evaluate(tree, repl_env))
            except SnekError as e:
                print("  error> {}".format(e))
        print()
