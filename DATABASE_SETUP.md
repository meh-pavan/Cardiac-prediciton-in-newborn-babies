## MongoDB Atlas Configuration

This project is configured to use MongoDB Atlas with the following connection string:

```
mongodb+srv://pavanmacha46:qwer1234@cluster0.smjhu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
```

This connection string is already set up in the `.env` file. No additional configuration is needed for the database connection.

## Verifying the Connection

To verify that your application can connect to MongoDB Atlas:

1. Make sure you have internet connectivity
2. Run the following test script:

```python
# test_db_connection.py
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
```

Run the script with:

```bash
python test_db_connection.py
```

## Troubleshooting Atlas Connection

If you encounter connection issues:

1. **Network Issues**: Ensure your network allows outbound connections to MongoDB Atlas (port 27017)
2. **IP Whitelist**: Make sure your IP address is whitelisted in the MongoDB Atlas Network Access settings
3. **Credentials**: Verify the username and password in the connection string
4. **Database Name**: Ensure the database name in the `.env` file is correct

## Database Management with MongoDB Atlas

### Accessing the Atlas Dashboard

1. Go to [MongoDB Atlas](https://cloud.mongodb.com)
2. Sign in with the credentials for the account that owns the cluster
3. Select the cluster "Cluster0"
4. Use the "Collections" button to browse and manage your data

### Backing Up Data

MongoDB Atlas provides automated backups. To create a manual backup:

1. In the Atlas dashboard, go to your cluster
2. Click "..." (more options)
3. Select "Download Data"
4. Follow the instructions to export your data

### Restoring Data

To restore data to MongoDB Atlas:

```bash
mongorestore --uri="mongodb+srv://pavanmacha46:qwer1234@cluster0.smjhu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" --db cardiac_prediction ./backup/cardiac_prediction
```