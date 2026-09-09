# Python code style

These conventions apply to Python code maintained in this repository, including
research scripts. EditorConfig supplies basic editor settings. The remaining
conventions are followed when writing and reviewing code, without repository
formatters, linters, type checkers, custom validation scripts, or required extensions.

## Formatting and imports

- Indent with **2 spaces**, never tabs.
- Use a **127-character line limit**. Wrap code, comments, and prose manually
  where needed; unbreakable URLs may exceed the limit.
- Prefer **single quotes** for ordinary strings. Double quotes are acceptable
  to avoid unnecessary escaping. Use triple double quotes for docstrings.
- Use UTF-8, LF line endings, a final newline, and no trailing whitespace in
  Python files.
- Keep imports sorted, with standard-library, third-party, and local imports
  grouped separately. Use your editor's built-in import organization if
  available; otherwise sort them manually. Absolute and relative imports are
  both allowed.
- Naming is left to the author, including mathematical variable names.
- Do not use em or en dashes in code comments, docstrings, or documentation.
  Use ordinary hyphens or rephrase the sentence.

The basic editor settings live in [`.editorconfig`](../.editorconfig).

## Docstrings: reStructuredText

**Docstrings must be valid reStructuredText (reST).** Use Sphinx field lists when
documenting parameters, return values, or exceptions. Use double backticks for
inline code and reST directives or literal blocks for examples.

Whether to add a docstring, and which details to document, is left to the author.
Plain descriptions and omitted docstrings are allowed. There is no documentation
coverage requirement, and types do not need to be repeated from annotations.

```python
def residual(score: float, target_mean: float) -> float:
  """Remove the training-set target mean.

  :param score: Complexity score for an image-target pair.
  :param target_mean: Target mean computed on the training split.
  :returns: The residualised complexity score.
  """
  return score - target_mean
```

Use reST field lists rather than Google/NumPy section headers. For an examples
heading, use `.. rubric:: Examples`. Docstring syntax is checked during review;
there is no automatic validator.

## Types

Use explicit parameter and return annotations on our functions and methods,
including private helpers. Keep typing as strict as possible within our own code:
use concrete types, typed containers, and precise interfaces. This is a coding
convention, with no required type-checking tool.

Third-party typing limitations should be contained at the library boundary:

- Prefer available stubs, typed wrappers, or narrow protocols.
- Where the library cannot express a known type, use a justified `cast` or a
  small, explicitly documented use of `Any` at the boundary. Return a precise
  type to our own code.
- Keep any necessary relaxation local and explain the library limitation.
  Avoid allowing untyped values to spread through the research implementation.

## Editor support

Compatible editors can apply the indentation, encoding, line-ending, final-newline,
and trailing-whitespace settings in `.editorconfig`. The file also records the
127-character line limit; whether it displays a ruler or affects wrapping depends
on the editor. It does not guarantee that every line meets the limit.

Quotes, reST docstrings, typing, naming, and import organization remain documented
conventions. There are no style dependencies to install or validation commands to run.
