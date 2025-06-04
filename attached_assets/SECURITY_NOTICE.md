# Security Notice for Attached Assets

⚠️ **IMPORTANT SECURITY WARNING** ⚠️

The files in this directory (`attached_assets/`) contain legacy code that **SHOULD NOT BE USED** in production environments due to significant security vulnerabilities.

## Security Issues in Legacy Code

The `app.py` file in this directory contains:

1. **Information Disclosure**: Raw error messages exposed to users
2. **Sensitive Data Logging**: API details logged to console/server logs
3. **Environment Variable API Keys**: Insecure API key handling from environment
4. **Insufficient Input Validation**: No proper input sanitization
5. **Error Information Leakage**: Detailed error context exposed to users

## Use Main Application Instead

**Always use the main `app.py` in the root directory** which has been security-hardened with:

- ✅ Secure API key handling with cryptographic hashing
- ✅ Proper input validation and sanitization
- ✅ Sanitized error handling (no information disclosure)
- ✅ Secure session management
- ✅ Memory cleanup mechanisms
- ✅ No sensitive data logging

## Recommendation

**Delete or ignore the files in this directory** - they are for reference only and contain security vulnerabilities that could expose sensitive information.

---
*Security review completed on 2025-06-04*