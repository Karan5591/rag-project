"""
Script to clean and verify the PGvector database.
"""
from app.services.database import get_connection


def clean_database():
    """Drop and recreate the documents table."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            # Drop existing table
            cursor.execute("DROP TABLE IF EXISTS documents;")
            print("[OK] Dropped existing documents table")
        
        connection.commit()
    
    print("[OK] Database cleaned successfully")


def verify_database_empty():
    """Check if documents table is empty."""
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM documents;")
                count = cursor.fetchone()[0]
        
        if count == 0:
            print("[OK] Database is EMPTY - ready for fresh insertion")
            return True
        else:
            print(f"[ERROR] Database still has {count} documents")
            return False
    except Exception as e:
        print(f"[WARN] Table doesn't exist yet: {e}")
        return True  # Table doesn't exist = empty


def show_table_info():
    """Display information about the documents table."""
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                # Get table info
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'documents'
                    ORDER BY ordinal_position;
                """)
                columns = cursor.fetchall()
                
                if columns:
                    print("\n[TABLE] Documents Table Schema:")
                    for col_name, col_type in columns:
                        print(f"   - {col_name}: {col_type}")
                    
                    # Get row count
                    cursor.execute("SELECT COUNT(*) FROM documents;")
                    count = cursor.fetchone()[0]
                    print(f"\n[INFO] Total Documents: {count}")
                else:
                    print("\n[WARN] Documents table does not exist yet")
    except Exception as e:
        print(f"\n[WARN] Error reading table info: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("PGvector Database Cleaner")
    print("=" * 60)
    
    # Show current state
    print("\n[STEP 1] Checking current database state...")
    show_table_info()
    
    # Clean database
    print("\n[STEP 2] Cleaning database...")
    clean_database()
    
    # Verify
    print("\n[STEP 3] Verifying database is empty...")
    verify_database_empty()
    
    print("\n" + "=" * 60)
    print("[OK] Database is ready for fresh insertion!")
    print("=" * 60)
