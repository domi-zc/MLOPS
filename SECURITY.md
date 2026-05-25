# Security Policy

This document outlines our vulnerability reporting process and the secure development guidelines all contributors are expected to follow.

## Reporting a Vulnerability

If you discover a security vulnerability or potential risk in this repository, please **do not create a public issue**.

### How to Report

Please report any security concerns directly via email to:
**dominic.zybach@stud.hslu.ch**

To help us resolve the issue as quickly as possible, please include:
* A detailed description of the vulnerability.
* Step-by-step instructions to reproduce the issue.

### Our Process

1. We will acknowledge receipt of your vulnerability report as soon as possible.
2. The reported issue will then be investigated to develop an appropriate fix.
3. Once the vulnerability has been patched and merged into the `main` branch, we will notify you.

Thank you for helping keep this project safe!

## Secure Development Guidelines

To maintain the integrity of this project, we enforce strict rules around credential handling and cloud authentication. Contributors must adhere to the following practices:

* **No Hardcoded Secrets:** Never commit passwords, API keys, or any sensitive credentials to version control. 
* **Local Environments:** For local development, store disposable or environment-specific credentials in `.env` files. Ensure these files are properly excluded via your `.gitignore`.
* **CI/CD Configuration:** GitHub repository variables should be strictly reserved for non-sensitive configuration data, such as public identifiers or workflow inputs.
* **Cloud Authentication:** When setting up CI/CD pipelines, utilize GitHub's OpenID Connect (OIDC) for secure, short-lived cloud authentication rather than relying on static service account keys.
* **Production Secrets:** Shared cloud credentials or runtime secrets must be injected via a managed Secret Manager or the secure runtime environment itself. Do not store them in repository syncing mechanisms or committed code examples.
