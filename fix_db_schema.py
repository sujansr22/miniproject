
import MySQLdb

def fix_schema():
    try:
        # Connect to database
        db = MySQLdb.connect(
            host="localhost",
            user="root",
            passwd="Sujan@123",
            db="crop_prediction_db"
        )
        
        cursor = db.cursor()
        
        # Renaissance password column to password_hash
        print("Renaming 'password' column to 'password_hash'...")
        try:
            cursor.execute("ALTER TABLE users CHANGE COLUMN password password_hash VARCHAR(255) NOT NULL")
            db.commit()
            print("Successfully renamed column!")
        except Exception as e:
            print(f"Error renaming column: {e}")
            
        cursor.close()
        db.close()
        
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    fix_schema()
