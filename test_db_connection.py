from modules.database_module import get_db_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get database manager
db_manager = get_db_manager()

# Test connection
try:
    # List all collections in the database
    collections = db_manager.db.list_collection_names()
    print(f"Successfully connected to MongoDB Atlas!")
    print(f"Collections in database: {collections}")
    
    # Test inserting a document
    test_result = db_manager.insert_patient_data({
        "patient_id": "TEST_CONNECTION",
        "test_field": "This is a test document"
    })
    print(f"Successfully inserted test document with ID: {test_result}")
    
    print("Database connection test completed successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")
    import traceback
    print(traceback.format_exc())