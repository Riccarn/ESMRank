# Releasing ESMRank to PyPI

Releases are published by the GitHub Actions workflow `.github/workflows/publish.yml`
using PyPI Trusted Publishing, so no API token is stored in GitHub.

## One-time setup

### 1. PyPI (and optionally TestPyPI)

On https://pypi.org (account with 2FA) go to **Account settings → Publishing** and add a
*pending publisher* (it lets you publish a project that does not exist yet):

| Field | Value |
|-------|-------|
| PyPI project name | `esmrank` |
| Owner | `gambalab` |
| Repository name | `ESMRank` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

For dry runs do the same on https://test.pypi.org with environment name `testpypi`
(it is a separate site with a separate account).

### 2. GitHub

In the repository go to **Settings → Environments** and create `pypi` (and `testpypi`).
For `pypi` it is a good idea to add *Required reviewers*, so that nothing is uploaded
until someone approves the run.

## Releasing a new version

1. Set the new version in `src/esmrank/__init__.py` (`__version__`) and merge it into `master`.
   The version is read from there by the build; a version that was uploaded once can never be
   uploaded again, so bump it for every release.
2. Optional dry run: **Actions → Publish to PyPI → Run workflow**. It runs the tests, builds the
   package and uploads it to TestPyPI.
3. **Releases → Draft a new release**, create the tag `vX.Y.Z` (it must match `__version__`,
   the workflow checks it) and publish the release. The workflow runs the tests, builds the
   package and uploads it to PyPI (after approval if you set required reviewers).

## After the first upload

- On the PyPI project page: **Manage → Collaborators**, add the other maintainers as Owners.
- In `README.md` the install command can become plain `pip install esmrank`.
