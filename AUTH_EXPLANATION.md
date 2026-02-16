# Authentication System - How It Works

## Overview

This authentication system provides a simple, secure login mechanism for the Smart Crop Prediction application. It uses MySQL for data storage, bcrypt for password hashing, and Flask sessions for user authentication.

---

## Components

### 1. Database Layer (MySQL)

**Table: `users`**
```sql
- id (INT, PRIMARY KEY, AUTO_INCREMENT)
- username (VARCHAR(50))
- email (VARCHAR(100), UNIQUE)
- password_hash (VARCHAR(255))
- created_at (TIMESTAMP)
```

**Purpose:**
- Stores user credentials securely
- Email is unique to prevent duplicate accounts
- Passwords are NEVER stored in plain text

---

### 2. Backend Layer (Flask)

**New Routes:**

#### `GET /login`
- Serves the login page HTML
- No authentication required

#### `POST /api/login`
- Accepts: `{ email, password }`
- Validates credentials against database
- Uses bcrypt to verify password hash
- Creates server-side session on success
- Returns user info (username, email)
- Client stores user info in localStorage for display

#### `GET /signup`
- Serves the signup page HTML
- No authentication required

#### `POST /api/signup`
- Accepts: `{ username, email, password }`
- Validates email format
- Checks if email already exists
- Hashes password using bcrypt
- Inserts new user into database
- Redirects to login page

#### `GET /logout`
- Clears server-side session
- Redirects to login page
- Client clears localStorage

#### `GET /forgot-password`
- Serves the forgot password page HTML

#### `POST /api/forgot-password`
- Accepts: `{ email, new_password }`
- Checks if email exists in database
- Hashes new password using bcrypt
- Updates password in database
- Redirects to login page

#### `GET /app` (Modified)
- Now checks if user is logged in
- Redirects to login if no session found
- Serves main application if authenticated

---

### 3. Frontend Layer (HTML + JavaScript)

#### `login.html`
- Email and password input fields
- Client-side form validation
- Sends POST request to `/api/login`
- Stores user info in localStorage on success
- Redirects to `/app`

#### `signup.html`
- Username, email, password, confirm password fields
- Client-side password matching validation
- Sends POST request to `/api/signup`
- Redirects to login on success

#### `forgot_password.html`
- Email, new password, confirm password fields
- Client-side password matching validation
- Sends POST request to `/api/forgot-password`
- Redirects to login on success

#### `app.html` (Modified)
- Added logout button in navigation
- Displays username from localStorage
- Logout button clears localStorage and redirects

---

## Security Features

### 1. Password Hashing
```python
# During signup/password reset:
password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# During login:
bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
```

**Why bcrypt?**
- Industry-standard password hashing
- Includes salt automatically
- Computationally expensive (prevents brute force)
- One-way encryption (cannot be reversed)

### 2. Session Management
```python
# Create session on login:
session['user_id'] = user['id']
session['user_email'] = user['email']
session['username'] = user['username']

# Check session before serving protected pages:
if 'user_email' not in session:
    return redirect(url_for('login_page'))

# Clear session on logout:
session.clear()
```

**Why sessions?**
- Server-side storage (more secure than cookies)
- Automatic expiration
- Cannot be tampered with by client
- Built into Flask

### 3. Input Validation

**Backend:**
- Required field checks
- Email format validation
- Email uniqueness check
- SQL injection prevention (parameterized queries)

**Frontend:**
- HTML5 input validation
- Password matching check
- Minimum password length
- User-friendly error messages

---

## Data Flow Diagrams

### Signup Flow
```
User fills form
    ↓
Client validates (password match, length)
    ↓
POST /api/signup { username, email, password }
    ↓
Backend validates (required fields, email format)
    ↓
Check if email exists in database
    ↓
Hash password with bcrypt
    ↓
INSERT INTO users (username, email, password_hash)
    ↓
Return success
    ↓
Redirect to login page
```

### Login Flow
```
User enters credentials
    ↓
POST /api/login { email, password }
    ↓
Backend validates (required fields)
    ↓
SELECT user FROM users WHERE email = ?
    ↓
Verify password with bcrypt.checkpw()
    ↓
Create session (user_id, email, username)
    ↓
Return user info
    ↓
Store in localStorage (for UI display only)
    ↓
Redirect to /app
```

### Protected Page Access
```
User navigates to /app
    ↓
Backend checks: 'user_email' in session?
    ↓
NO → Redirect to /login
YES → Serve app.html
    ↓
JavaScript reads username from localStorage
    ↓
Display "Welcome, [username]!" in nav
```

### Logout Flow
```
User clicks Logout button
    ↓
GET /logout
    ↓
Backend: session.clear()
    ↓
Redirect to /login
    ↓
(Optionally: JavaScript clears localStorage)
```

### Forgot Password Flow
```
User enters email and new password
    ↓
Client validates (password match, length)
    ↓
POST /api/forgot-password { email, new_password }
    ↓
Backend validates (required fields)
    ↓
SELECT user FROM users WHERE email = ?
    ↓
If not found → Return error
    ↓
Hash new password with bcrypt
    ↓
UPDATE users SET password_hash = ? WHERE email = ?
    ↓
Return success
    ↓
Redirect to login page
```

---

## Why This Approach?

### ✅ Advantages

1. **Simple & Beginner-Friendly**
   - No complex OAuth flows
   - No external dependencies
   - Easy to understand and modify

2. **Secure**
   - Passwords are hashed (never stored in plain text)
   - Sessions are server-side
   - SQL injection protected
   - No sensitive data in localStorage

3. **Local & Offline**
   - Works completely offline
   - No cloud services required
   - No API keys needed
   - No costs

4. **Clone & Run Ready**
   - Just setup MySQL and run
   - No configuration files needed
   - Clear setup instructions

### ⚠️ Limitations

1. **No Email Verification**
   - Users can register with any email
   - No email confirmation required

2. **No Password Recovery via Email**
   - Forgot password doesn't send emails
   - User must know their registered email

3. **No Rate Limiting**
   - No protection against brute force
   - Can be added with Flask-Limiter

4. **No Multi-Factor Authentication**
   - Single-factor (password only)

5. **Session Storage**
   - Sessions stored in memory (lost on restart)
   - Can be improved with Redis/database sessions

---

## localStorage vs Session

### What's in localStorage?
```javascript
localStorage.setItem('userEmail', 'user@example.com');
localStorage.setItem('username', 'JohnDoe');
```

**Purpose:** Display user info in UI only
**Security:** Not used for authentication
**Cleared:** On logout

### What's in Session?
```python
session['user_id'] = 123
session['user_email'] = 'user@example.com'
session['username'] = 'JohnDoe'
```

**Purpose:** Server-side authentication
**Security:** Cannot be tampered with
**Cleared:** On logout or server restart

---

## Common Questions

### Q: Can users see their password?
**A:** No. Passwords are hashed immediately and cannot be reversed.

### Q: What if I forget my password?
**A:** Use the "Forgot Password" feature. You'll need to know your registered email.

### Q: Is the password stored in localStorage?
**A:** No. Only username and email (for display) are stored. Never passwords.

### Q: Can someone steal my session?
**A:** Sessions are server-side and use secure cookies. As long as you're on HTTPS in production, sessions are safe.

### Q: What happens if MySQL is down?
**A:** The app will show database connection errors. Users cannot login until MySQL is back up.

### Q: Can I use this in production?
**A:** Yes, but add:
- HTTPS (SSL certificate)
- Environment variables for credentials
- Rate limiting
- Email verification
- Better session storage (Redis)
- Logging and monitoring

---

## Extending the System

### Add Email Verification
1. Add `email_verified` column to users table
2. Generate verification token on signup
3. Send email with verification link
4. Verify token and update `email_verified`

### Add Password Strength Requirements
```javascript
// Client-side
function validatePassword(password) {
    const minLength = 8;
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*]/.test(password);
    
    return password.length >= minLength && 
           hasUpperCase && hasLowerCase && 
           hasNumbers && hasSpecialChar;
}
```

### Add Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    # ... existing code
```

### Add Remember Me
```python
# On login:
if request.json.get('remember_me'):
    session.permanent = True
    app.permanent_session_lifetime = timedelta(days=30)
```

---

## Conclusion

This authentication system provides a solid foundation for securing your crop prediction application. It's simple, secure, and easy to extend. Follow the SETUP_GUIDE.md for installation instructions.
