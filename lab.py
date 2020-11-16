#!/usr/bin/env python3
"""6.009 Lab 9: Snek Interpreter"""

import doctest
import sys
TURTLE_ENABLED = False
MULTIEXP_ENABLED = False
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
    def __init__(self, message='', incomplete=True):
        """If incomplete = True, this error represents an incomplete expression, rather than a malformed one."""
        super().__init__(message)
        self.incomplete = incomplete


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

def is_num_char(x):
    """Checks if x is a digit"""
    return ord('0') <= ord(x) <= ord('9')

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
                while idx < len(source) and is_num_char(source[idx]):
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
            elif c == '`':
                yield '`'
            elif c == "'":
                yield "'"
            elif c == ',':
                idx += 1
                if idx < len(source) and source[idx] == '@':
                    yield ',@'
                else:
                    idx -= 1
                    yield ','
            elif c == ';': # Handle comments, by skip to the newline, then the loop consumes the newline
                while idx < len(source) and source[idx] != '\n':
                    idx += 1
            elif is_num_char(c):
                yield parse_number()
            elif c == '-':
                idx += 1
                if idx < len(source) and is_num_char(source[idx]):
                    yield '-' + parse_number()
                else:
                    idx -= 1
                    yield '-'
            elif c == '#':
                idx += 1
                if idx < len(source) and (source[idx] == 't' or source[idx] == 'T'):
                    yield '#t'
                elif idx < len(source) and (source[idx] == 'f' or source[idx] == 'F'):
                    yield '#f'
                else:
                    idx -= 1
                    yield '#'
            else:
                yield parse_symbol()
            idx += 1
    return list(filter(lambda x: not not x, _tokenize(source)))

KEYWORDS = ["define", "lambda", "let", "letrec", "set!", "quote", "unquote", "unquote-splicing", "quasiquote", "turtle", "if", "and", "or"]
def check_set_form(expr):
    """Checks the set form of an S expression, and raises SnekSyntaxErrors when failing"""
    def check_set_form_length(name, leng, exact=True):
        if (len(expr) != leng and exact) or (len(expr) < leng and not exact):
            raise SnekSyntaxError(incomplete=False, message="wrong set form for {}: expected {}{}, got {} arguments".format(name, "+" if not exact else "", leng - 1, len(expr) - 1))
    if expr:
        if expr[0] == 'define':
            check_set_form_length("define", 3, not MULTIEXP_ENABLED)
            if not (isinstance(expr[1], str) or (isinstance(expr[1], list) and len(expr[1]) > 0 and (all(map(lambda i: isinstance(i, str), expr[1]))))):
                raise SnekSyntaxError("malformed define", incomplete=False)
            if isinstance(expr[1], list) and expr[1][0] in KEYWORDS:
                raise SnekSyntaxError("cannot define keyword {}".format(expr[1][0]), incomplete=False)
        elif expr[0] == 'lambda':
            check_set_form_length("lambda", 3, not MULTIEXP_ENABLED)
            if not (isinstance(expr[1], list) and all(map(lambda i: isinstance(i, str), expr[1]))):
                raise SnekSyntaxError("malformed lambda", incomplete=False)
        elif expr[0] == 'if':
            check_set_form_length("if", 4)
        elif expr[0] == 'let' or expr[0] == 'letrec':
            check_set_form_length(expr[0], 3, not MULTIEXP_ENABLED)
            if not (isinstance(expr[1], list) and all(map(lambda i: isinstance(i, list) and len(i) == 2 and isinstance(i[0], str), expr[1]))):
                raise SnekSyntaxError("malformed {}".format(expr[0]), incomplete=False)
        elif expr[0] == 'set!':
            check_set_form_length("set!", 3)
            if not isinstance(expr[1], str):
                raise SnekSyntaxError("set! must be a symbol followed by a value", incomplete=False)
        elif expr[0] == 'quote':
            check_set_form_length("quote", 2)
        elif expr[0] == 'unquote':
            check_set_form_length("unquote", 2)
        elif expr[0] == 'quasiquote':
            check_set_form_length("quasiquote", 2)
        elif expr[0] == 'unquote-splicing':
            check_set_form_length("unquote-splicing", 2)
        elif expr[0] == 'turtle':
            check_set_form_length("turtle", 2, False)

QUOTEEXPR = ["quote", "quasiquote"]
def parse(tokens, complete=True):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
        complete (flag): if True, will error if all tokens are not consume
    """
    # Check first token, it can be one of 3 things if it is valid:
    # - '(' for S-expressions
    # - a string beginning with a digit for numbers
    # - a string beginning with a non-digit for symbols
    def parse_item(tokens, set_expr=True):
        if tokens:
            token = tokens.pop(0)
            if token == '(':
                # Parse out S-expression
                expr = []
                while len(tokens) > 0 and tokens[0] != ')':
                    if expr and expr[0] in QUOTEEXPR:
                        item = parse_item(tokens, set_expr=False)
                    else:
                        item = parse_item(tokens, set_expr=set_expr)
                    expr.append(item)
                if len(tokens) > 0 and tokens[0] == ')':
                    tokens.pop(0)
                else:
                    raise SnekSyntaxError
                
                if set_expr: # If expr is not empty
                    check_set_form(expr)
                return expr
            elif is_num_char(token[0]) or (len(token) > 1 and token[0] == '-' and is_num_char(token[1])):
                try:
                    # Parse a number from the token
                    if sum(map(lambda c: 1 if c == '.' else 0, token)) == 1: # We have exactly 1 decimal point, a valid float
                        return float(token)
                    elif sum(map(lambda c: 1 if c == '.' else 0, token)) == 0: # We have 0 decimal points, a valid int
                        return int(token)
                    else: # Some weird numerical identifier
                        return token
                except ValueError as e: # Something went wrong in parsing, treat as identifier
                    print(e)
                    return token
            elif token == '#t':
                return True
            elif token == '#f':
                return False
            elif token == 'nil':
                return Nil()
            elif token == '`':
                item = parse_item(tokens, False)
                return ["quasiquote", item]
            elif token == "'":
                item = parse_item(tokens, False)
                return ["quote", item]
            elif token == ',':
                item = parse_item(tokens)
                return ["unquote", item]
            elif token == ',@':
                item = parse_item(tokens)
                return ["unquote-splicing", item]
            elif token == ')': # Expected a expression, got an expression ender
                raise SnekSyntaxError(incomplete=False, message="unexpected ), expected literal or (")
            else:
                return token
        else:
            raise SnekSyntaxError # Expected somthing, got nothing
    top_level = parse_item(tokens)
    if len(tokens) != 0 and complete:
        raise SnekSyntaxError # We expected all tokens consumed, but this didn't happen, so something went wrong
    return top_level

def list_typecheck(val, name, msg):
    """Typechecks for a list cons"""
    if type(val) != Pair and val != Nil():
        raise SnekEvaluationError(name + " error: " + msg)

######################
# Built-in Functions #
######################

def product(*args):
    v = 1
    for i in args:
        v *= i
    return v

def division(*args):
    x = args[0]
    for i in args[1:]:
        x = (x / i)
    return x

def all_equal(*args): # =?
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
    def comp(*args):
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

def not_fn(*args):
    if len(args) != 1:
        raise SnekEvaluationError("not can only take 1 argument")
    return not args[0]

def cons(*args):
    if len(args) != 2:
        raise SnekEvaluationError("cons can only take 2 arguments")
    return Pair(args[0], args[1], list_mode=False)

def car(*args):
    if len(args) != 1 or not isinstance(args[0], Pair):
        raise SnekEvaluationError("car can only take 1 cons argument")
    item = args[0]
    return item.car

def cdr(*args):
    if len(args) != 1 or not isinstance(args[0], Pair):
        raise SnekEvaluationError("cdr can only take 1 cons argument")
    item = args[0]
    return item.cdr

def list_snek(*args):
    if not args:
        return Nil()
    else:
        top = Pair(args[0], Nil())
        l = top
        for arg in args[1:]:
            i = Pair(arg, Nil())
            l.cdr = i
            l = i
        return top

def length(*args):
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

def elt_at_index(*args):
    if len(args) != 2 or not (isinstance(args[0], Pair) and isinstance(args[1], int)) or args[1] < 0:
        raise SnekEvaluationError("elt-at-index can only take 2 arguments: a list cons and positive integer")
    n = args[1]
    l = args[0]
    while n > 0:
        n -= 1
        if l == Nil():
            break
        l = l.cdr
        list_typecheck(l, "elt-at-index", "Cannot operate on non-list cons")
    if l != Nil():
        return l.car
    else:
        raise SnekEvaluationError("index {} out of bounds for {}".format(args[1], args[0]))

def concat(*args):
    if not args:
        return Nil()
    else:
        [list_typecheck(lst, "concat", "Can only join list cons") for lst in args] # Typecheck all the lists
        lists = list(args)
        flst = None # First cons of list to return
        nlst = None # Current end of flst
        while lists:
            clst = lists.pop(0).clone()
            # print(flst, nlst, clst)
            if clst != Nil():
                if not flst:
                    flst = clst
                    nlst = clst
                else:
                    nlst.cdr = clst
            else:
                continue
            while nlst and nlst.cdr != Nil():
                nlst = nlst.cdr
                list_typecheck(nlst, "concat", "Can only join list cons")
        if flst:
            return flst
        else:
            return Nil()

def map_snek(*args):
    if len(args) != 2 or not (isinstance(args[1], Pair) or args[1] == Nil()):
        raise SnekEvaluationError
    flst = args[1].clone()
    clst = flst
    while clst != Nil():
        try:
            clst.car = args[0](clst.car)
        except TypeError as e:
            raise SnekEvaluationError("Could not call arg 0 as function: {}".format(e))
        it = clst.cdr.clone()
        clst.cdr = it
        clst = it
        list_typecheck(clst, "map", "Can only map list cons")
    return flst

def filter_snek(*args):
    if len(args) != 2 or not (isinstance(args[1], Pair) or args[1] == Nil()):
        raise SnekEvaluationError
    plst = args[1] # Points to query of list
    flst = None # Points to begin of returned list
    clst = None # Points to current passed of returned list
    while plst != Nil():
        try:
            cond = args[0](plst.car)
        except TypeError as e:
            raise SnekEvaluationError("Could not call arg 0 as function: {}".format(e))
        if cond:
            it = plst.clone()
            if not flst:
                flst = it
                clst = it
            else:
                clst.cdr = it
                clst = it
        plst = plst.cdr
        list_typecheck(plst, "filter", "Can only filter list cons")
    if clst and clst != Nil(): # Terminate the new list
        clst.cdr = Nil()

    if flst:
        return flst
    else:
        return Nil()

def reduce_snek(*args):
    if len(args) != 3 or not (isinstance(args[1], Pair) or args[1] == Nil()):
        raise SnekEvaluationError
    val = args[2]
    cons = args[1]
    while cons != Nil():
        try:
            val = args[0](val, cons.car)
            cons = cons.cdr
            list_typecheck(cons, "reduce", "Only supports list cons, not arbitrary cons")
        except TypeError as e:
            raise SnekEvaluationError("Could not call arg 0 as function: {}".format(e))
    return val

def begin_snek(*args):
    return args[-1]

def int_snek(*args):
    if len(args) != 1:
        raise SnekEvaluationError("int can only take 1 argument")
    return int(args[0])

def set_car_mut(*args):
    if len(args) != 2 or not isinstance(args[0], Pair):
        raise SnekEvaluationError("set-car! can only take 2 arguments: a cons and a value")
    args[0].car = args[1]
    return args[1]

def import_snek(*args):
    if len(args) != 1 or not isinstance(args[0], str):
        raise SnekEvaluationError("py-import only accepts a string/symbol quote")
    return __import__(args[0])

def getattr_snek(*args):
    # We can't actually check if for the module class directly (cuz idk how), but we do know that sys is of that type
    # So we just check if it and sys have the same type...
    if len(args) != 2 or not isinstance(args[0], type(sys)) or not isinstance(args[1], str):
        raise SnekEvaluationError("getattr can only get 2 arguments: a module, and a string/symbol quote")
    return getattr(args[0], args[1])

def is_num(*args):
    if len(args) != 1:
        raise SnekEvaluationError("num? only accepts 1 argument")
    return type(args[0]) == int or type(args[0]) == float

def is_list(*args):
    if len(args) != 1:
        raise SnekEvaluationError("list? only accepts 1 argument")
    l = args[0]
    while isinstance(l, Pair) or isinstance(l, Nil):
        if l == Nil():
            return True
        else:
            l = l.cdr
    return False # If we reach here without returning early, l is not Pair and not nil, so it is not a list

snek_builtins = {
    '+': lambda *args: sum(args),
    '-': lambda *args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
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
    'set-car!': set_car_mut,
    'py-import': import_snek,
    'getattr': getattr_snek,
    'num?': is_num,
    'list?': is_list,
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

    def close(self):
        """
        Redefines all parent elements in the current scope to the value in the parent scope (to prevent changes in parent scope from affecting current scope)
        NOTE: bc of Python aliasing, might not fully protect against changes. If changes occur, check other code.
        """
        in_scope = self.defined_names()
        all_scope = self.defined_names(tree=True)
        for name in all_scope:
            if name not in in_scope:
                self.define(name, self.lookup(name))

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

    def __call__(self, *args):
        """
        Call this function.
        Creates a new environment, and expects *args to be evaluated values.
        """

        func_env = Environment(self.parent)
        self.define_args(func_env, *args)
        return evaluate(self.body, func_env)

    def define_args(self, env, *args):
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
    def __init__(self, car=None, cdr=None, list_mode=True):
        self.car = car
        self.cdr = cdr
        self.list_mode = list_mode

    def __repr__(self):
        return "[{} {}]".format(repr(self.car), repr(self.cdr))

    def __eq__(self, o):
        i = self
        while not isinstance(i, Nil):
            if isinstance(o, Pair) and isinstance(i, Pair):
                if i.car != o.car:
                    return False
                else:
                    i = i.cdr
                    o = o.cdr
            else:
                return False
        return isinstance(i, Nil) and isinstance(o, Nil)

    def __str__(self):
        if self.list_mode:
            string = "("
            item = self
            while item != Nil():
                string += "{}{}".format(" " if string != '(' else "", item.car)
                item = item.cdr
                list_typecheck(item, "__str__", "INTERP ERROR: list_mode cons must be a list")
            string += ")"
            return string
        else:
            return "[{} {}]".format(self.car, self.cdr)

    def clone(self):
        """Creates an independant copy of this Pair"""
        return Pair(self.car, self.cdr, self.list_mode)

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

def quasiquote(datum, env):
    """
    Processes a template, to produce a quasiquoted template
    datum(tree) - datum to quasiquote
    env(Environment) - environment to evaluate in
    level(int, optional) - the level of quasiquote this expression is at.
        Plain S-expressions (so non-special forms), are only evaluated if this is == to 0, otherwise, it will directly return the list.
    """
    # We use this method, to:
    #   1. override 'unquote' and introduce 'unquote-splicing' in the context of quasiquote
    #   2. Recusively detect 'unquote*' expressions in lists
    if isinstance(datum, list):
        def process_splice_list(itr):
            """
            Merges a list of (item, splice_flag) into a Python list, where:
            item - any valid external representation
            splice_flag - if True, item must be a list, and it will be spliced into the Python list
            """
            quote = []
            for (item, splice) in itr:
                # print(item, splice)
                if not splice:
                    quote.append(item)
                else:
                    while item != Nil():
                        quote.append(item.car)
                        item = item.cdr
                        list_typecheck(item, "unquote-splicing", "attempted to splice non-list cons")
            return quote

        def process_qquote_items(item, level=1):
            """
            Process quasiquoted items.
            item(tree) - item to quasiquote
            level(int, optional) - the level of quasiquoting. Expressions are only evaluated if the level <= 0
            """
            nonlocal env
            if isinstance(item, list):
                if len(item) < 1:
                    return ([], False)
                if item[0] == 'quasiquote':
                    return (list_snek(*process_splice_list(map(lambda it: process_qquote_items(it, level + 1), item))), False)
                elif item[0] == 'unquote':
                    if level == 1:
                        val, splice = process_qquote_items(item[1], level - 1)
                        return (val, False)
                    else:
                        return (list_snek(*process_splice_list(map(lambda it: process_qquote_items(it, level - 1), item))), False)
                elif item[0] == 'unquote-splicing':
                    if level == 1:
                        val, splice = process_qquote_items(item[1], level - 1)
                        list_typecheck(val, "unquote-splicing", "attempted to splice non-list cons")
                        return (val, True)
                    else:
                        return (list_snek(*process_splice_list(map(lambda it: process_qquote_items(it, level - 1), item))), False)
                else:
                    it = None
                    if level <= 0:
                        it = evaluate(item, env)
                    else:
                        it = list_snek(*process_splice_list(map(lambda it: process_qquote_items(it, level), item)))
                    return (it, False)
            elif isinstance(item, str):
                it = None
                if level <= 0:
                    it = evaluate(item, env)
                else:
                    it = item
                return (it, False)           
            else:
                return (item, False)
        ret, splice = process_qquote_items(datum)
        # Ignore splice at top-level, because there is nothing to splice into...
        return ret
    else:
        return datum



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

    loopc = 0 # Loop counter
    loopt = 1 # Loop target (how many times we want to loop)
    def tailcall(code, _env=None):
        """Tail-calls the given code, by replacing the tree and then increasing the loop target"""
        nonlocal loopt, tree, env
        if _env:
            env = _env
        tree = code
        loopt += 1
    def multibody(body):
        """Takes a multiexpression body, and converts it into a single expression"""
        if len(body) > 1:
            return ["begin"] + body
        else:
            return body[0]
    while loopc < loopt:
        if isinstance(tree, str):
            return env.lookup(tree)
        elif isinstance(tree, int) or isinstance(tree, float) or isinstance(tree, bool) or isinstance(tree, Nil):
            return tree  
        elif isinstance(tree, list): # S-expression
            if len(tree) < 1:
                raise SnekEvaluationError
            try:
                check_set_form(tree)
            except SnekSyntaxError as e:
                raise SnekEvaluationError("Metaprogramming error: {}".format(e))
            if tree[0] == 'define': # Definition time
                name = None
                value = None
                body = multibody(tree[2:])
                if isinstance(tree[1], str):
                    name = tree[1]
                    value = evaluate(body, env)
                elif isinstance(tree[1], list):
                    name = tree[1][0]
                    params = tree[1][1:]
                    value = UserFunction(env, params, body)
                env.define(name, value)
                return value
            elif tree[0] == 'lambda': # Lambda time
                params = tree[1]
                body = multibody(tree[2:])
                return UserFunction(env, params, body)
            elif tree[0] == 'if':
                cond = evaluate(tree[1], env)
                if cond:
                    tailcall(tree[2])
                else:
                    tailcall(tree[3])
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
                body = multibody(tree[2:])
                tailcall(body, let_env)
            elif tree[0] == 'letrec':
                # Create a shared environment for execution, and first define all names to None (so they exist, but aren't actually usable)
                pairs = tree[1]
                let_env = Environment(env)
                for pair in pairs:
                    let_env.define(pair[0], None)
                for pair in pairs:
                    let_env.define(pair[0], evaluate(pair[1], let_env))
                body = multibody(tree[2:])
                tailcall(body, let_env)
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
            elif tree[0] == 'quote':
                def quoter(datum):
                    if isinstance(datum, list):
                        return list_snek(*[quoter(it) for it in datum])
                    else:
                        return datum
                return quoter(tree[1])
            elif tree[0] == 'unquote':
                datum = tree[1]
                def unquoter(datum):
                    """Attempts to unquote data"""
                    if isinstance(datum, Pair):
                        utree = []
                        def snek_to_py(lst):
                            if isinstance(lst, Pair):
                                list_typecheck(datum, "unquote", "attempted to unquote a non-lost cons")
                                nl = []
                                while lst != Nil():
                                    nl.append(snek_to_py(lst.car))
                                    lst = lst.cdr
                                    list_typecheck(datum, "unquote", "attempted to unquote a non-lost cons")
                                return nl
                            else:
                                return lst
                        # print(repr(snek_to_py(datum)))
                        return evaluate(snek_to_py(datum), env)
                    elif isinstance(datum, str) or isinstance(datum, list):
                        val = evaluate(datum, env)
                        return unquoter(val)
                    else:
                        return datum
                return unquoter(datum)
            elif tree[0] == 'quasiquote':
                return quasiquote(tree[1], env)
            else: # Environment call
                target = evaluate(tree[0], env)
                args = list(map(lambda subn: evaluate(subn, env), tree[1:]))
                try:
                    if type(target) is UserFunction:
                        # Do some tail-calling
                        nenv = Environment(target.parent)
                        target.define_args(nenv, *args)
                        tailcall(target.body, nenv)
                    else:
                        return target(*args)
                except TypeError as e: # Cannot call target as a function
                    raise SnekEvaluationError(e)
        else:
            raise SnekEvaluationError("cannot evaluate value {} of type {}".format(repr(tree), type(tree))) # Unexpected type encountered
        loopc += 1

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
    tree = None
    if MULTIEXP_ENABLED:
        trees = []
        while tokens:
            trees.append(parse(tokens, False))
        tree = ["begin"] + trees
    else:
        tree = parse(tokens)
    return evaluate(tree, env)

if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()

    repl_env = Environment(builtin_env)
    MULTIEXP_ENABLED = True # Enable multi expressions for define, let and lambda

    for arg in sys.argv[1:]:
        evaluate_file(arg, repl_env)

    while True:
        string = input("in> ")
        if string.lower() == "quit":
            break
        else:
            try:
                if not string:
                    continue # Jump back to loop start if string is empty
                tokens = tokenize(string)
                trees = []
                while not trees:
                    try:
                        # print(tokens)
                        tken = tokens[:]
                        while tken:
                            trees.append(parse(tken, False))
                    except SnekSyntaxError as e:
                        if e.incomplete and tokens and tokens[0] != '' and (tokens[0] == '(' or is_alpha(tokens[0][0]) or is_num(tokens[0][0])):
                            cont = input("..  ")
                            tokens += tokenize(cont)
                            trees = []
                        else:
                            # Cannot start an expression, error
                            raise e
                # print(trees)
                for tree in trees:
                    print("  out>", evaluate(tree, repl_env))
            except SnekError as e:
                print("  error> {}".format(e))
        print()
