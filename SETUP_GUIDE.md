# Authentication System Setup Guide

## Prerequisites

Before setting up the authentication system, ensure you have:
- Python 3.7 or higher installed
- MySQL Server installed and running
- Git (if cloning from repository)

---

## Step 1: Install MySQL

### Windows:
1. Download MySQL Installer from [https://dev.mysql.com/downloads/installer/](https://dev.mysql.com/downloads/installer/)
2. Run the installer and choose "Developer Default"
3. Set a root password during installation (remember this!)
4. Complete the installation

### Verify MySQL is running:
```bash
mysql --version
```

---

## Step 2: Create Database

1. Open MySQL Command Line Client or any MySQL GUI tool (like MySQL Workbench)
2. Login with your root credentials
3. Run the following command:

```sql
CREATE DATABASE crop_prediction_db;
```

---

## Step 3: Create Users Table

1. Navigate to your project directory
2. Run the SQL script to create the users table:

```bash
mysql -u root -p crop_prediction_db < create_users_table.sql
```

Or manually execute the SQL:

```sql
USE crop_prediction_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

## Step 4: Install Python Dependencies

Install the required packages:

```bash
pip install flask-mysqldb bcrypt
```

Or if you have a requirements.txt:

```bash
pip install -r requirements.txt
```

---

## Step 5: Configure Database Connection

Open `app.py` and update the MySQL configuration (around line 15-20):

```python
# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # Change to your MySQL username
app.config['MYSQL_PASSWORD'] = 'your_password'  # Change to your MySQL password
app.config['MYSQL_DB'] = 'crop_prediction_db'
```

**IMPORTANT:** Replace `'your_password'` with your actual MySQL root password.

---

## Step 6: Run the Application

Start the Flask application:

```bash
python app.py
```

You should see output like:
```
* Running on http://127.0.0.1:5003
* Restarting with stat
* Debugger is active!
```

---

## Step 7: Test the Authentication System

### 7.1 Sign Up
1. Open your browser and go to `http://localhost:5003`
2. You'll be redirected to the login page
3. Click "Sign Up"
4. Fill in:
   - Username: `testuser`
   - Email: `test@example.com`
   - Password: `password123`
   - Confirm Password: `password123`
5. Click "Sign Up"
6. You should be redirected to the login page with a success message

### 7.2 Login
1. On the login page, enter:
   - Email: `test@example.com`
   - Password: `password123`
2. Click "Login"
3. You should be redirected to the main application (`/app`)
4. You should see "Welcome, testuser!" in the navigation bar

### 7.3 Use the Application
- Enter soil data and get crop predictions
- Your session is maintained while you use the app

### 7.4 Logout
1. Click the "Logout" button in the navigation bar
2. You should be redirected to the login page
3. Your session and localStorage are cleared

### 7.5 Forgot Password
1. On the login page, click "Forgot Password?"
2. Enter your email: `test@example.com`
3. Enter new password: `newpassword123`
4. Confirm new password: `newpassword123`
5. Click "Reset Password"
6. You should be redirected to login
7. Login with the new password

---

## Troubleshooting

### Issue: "Access denied for user 'root'@'localhost'"
**Solution:** Check your MySQL username and password in `app.py`

### Issue: "No module named 'flask_mysqldb'"
**Solution:** Run `pip install flask-mysqldb`

### Issue: "Can't connect to MySQL server"
**Solution:** Ensure MySQL service is running:
- Windows: Check Services (services.msc) for "MySQL" service
- Start it if it's stopped

### Issue: "Table 'users' doesn't exist"
**Solution:** Run the SQL script again:
```bash
mysql -u root -p crop_prediction_db < create_users_table.sql
```

### Issue: Page shows "Internal Server Error"
**Solution:** Check the Flask console for error messages. Common issues:
- Database connection failed
- Missing dependencies
- Incorrect database credentials

---

## Security Notes

1. **Change the Secret Key:** In `app.py`, change the secret key to a random string:
   ```python
   app.secret_key = 'your-unique-random-secret-key-here'
   ```

2. **Password Security:** Passwords are hashed using bcrypt before storage. Never store plain text passwords.

3. **Production Deployment:** 
   - Use environment variables for database credentials
   - Set `debug=False` in production
   - Use HTTPS for secure communication

---

## How It Works

### Signup Flow
1. User fills signup form
2. Client-side validates password match
3. Data sent to `/api/signup`
4. Backend checks if email exists
5. Password is hashed using bcrypt
6. User data saved to MySQL
7. Redirect to login page

### Login Flow
1. User enters email and password
2. Data sent to `/api/login`
3. Backend retrieves user by email
4. Password verified using bcrypt
5. Session created with user info
6. User info saved to localStorage (for display only)
7. Redirect to `/app`

### Logout Flow
1. User clicks logout button
2. Request sent to `/logout`
3. Session cleared on server
4. Redirect to login page
5. Client-side clears localStorage

### Forgot Password Flow
1. User enters email and new password
2. Data sent to `/api/forgot-password`
3. Backend checks if email exists
4. New password is hashed
5. Password updated in database
6. Redirect to login page

---

## File Structure

```
miniproject/
├── app.py                      # Main Flask application with auth routes
├── create_users_table.sql      # SQL script to create users table
├── login.html                  # Login page
├── signup.html                 # Signup page
├── forgot_password.html        # Forgot password page
├── app.html                    # Main application (protected)
├── welcome.html                # Welcome page
├── style.css                   # Styles
└── ... (other existing files)
```

---

## Next Steps

After successful setup:
1. Customize the UI to match your brand
2. Add email verification (optional)
3. Implement password strength requirements
4. Add rate limiting for login attempts
5. Set up proper logging for security events

---

## Support

If you encounter any issues not covered in this guide:
1. Check the Flask console for error messages
2. Verify MySQL is running and accessible
3. Ensure all dependencies are installed
4. Check that the database and table exist
5. Verify database credentials in `app.py`
