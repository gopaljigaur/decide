# Releasing

One release per finished change. A change is finished when its code, its tests,
its documentation and its changelog entry are all in, not when the first commit
of it passes CI.

Between 15 and 20 October 2026 this package shipped fifteen releases, several of
them correcting the one before. That is what this file exists to prevent.

## Steps

1. Land the whole change on `main`, green on CI.
2. Write one `CHANGELOG.md` entry that says what changed and what a user has to
   do differently. Not one entry per commit.
3. Bump the version in `pyproject.toml` and `decide/__init__.py`. Stay in
   `0.2.x` unless the public API genuinely breaks.
4. `gh release create vX.Y.Z` against `main`. Publishing is automatic from
   there: the `publish` workflow builds, runs `twine check`, and uploads to PyPI
   through trusted publishing.
5. Install the published version in a clean environment and exercise the thing
   that changed. The PyPI page can take a minute to catch up.

## Not a reason to release

A typo, a docs-only edit, a lint fix, or realising the last release named
something badly. Those wait for the next real change.
