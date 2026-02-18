from flask import Flask
from flask_mysqldb import MySQL
from config import Config
from extensions import mysql

app = Flask(__name__)
app.config.from_object(Config)
mysql.init_app(app)

def migrate():
    with app.app_context():
        try:
            cur = mysql.connection.cursor()
            
            # Make farmer_id nullable in questions
            print("Modifying questions table...")
            cur.execute("ALTER TABLE questions MODIFY farmer_id INT NULL")
            
            # Make consultant_id nullable in answers
            print("Modifying answers table...")
            cur.execute("ALTER TABLE answers MODIFY consultant_id INT NULL")
            
            mysql.connection.commit()
            cur.close()
            print("Successfully modified schema for guest contributions!")
        except Exception as e:
            print(f"Error during migration: {e}")

if __name__ == "__main__":
    migrate()
