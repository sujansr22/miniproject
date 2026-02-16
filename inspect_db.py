
import MySQLdb

def inspect_table():
    try:
        # Connect to database
        db = MySQLdb.connect(
            host="localhost",
            user="root",
            passwd="Sujan@123",
            db="crop_prediction_db"
        )
        
        cursor = db.cursor()
        
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE 'users'")
        result = cursor.fetchone()
        
        if not result:
            print("Table 'users' does not exist!")
            return

        # Describe table
        print("Table 'users' structure:")
        cursor.execute("DESCRIBE users")
        columns = cursor.fetchall()
        for col in columns:
            print(col)
            
        cursor.close()
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_table()
