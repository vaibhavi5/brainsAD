import pymongo
from pymongo.errors import ConnectionFailure, OperationFailure
import sys

def get_mongo_specs(ip, port=27017):
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(f"mongodb://{ip}:{port}/", serverSelectionTimeoutMS=5000)

        # Check if the connection was successful
        client.admin.command('ismaster')
    except ConnectionFailure:
        print("Server not available")
        return

    print(f"\n{'=' * 40}")
    print(f"MongoDB Specifications for {ip}:{port}")
    print(f"{'=' * 40}\n")

    try:
        # Get server info
        server_info = client.server_info()
        print(f"Server version: {server_info.get('version', 'Unknown')}")

        # Only print uptime if available
        if 'uptime' in server_info:
            print(f"Server uptime: {server_info['uptime'] // 3600} hours")
        else:
            print("Server uptime: Not available")

        # Get database names
        db_names = client.list_database_names()

        for db_name in db_names:
            db = client[db_name]

            print(f"\n{'-' * 40}")
            print(f"Database: {db_name}")
            print(f"{'-' * 40}")

            try:
                # Get database stats
                stats = db.command("dbStats")

                # Time created (approximated from oldest collection)
                oldest_time = None
                for coll_name in db.list_collection_names():
                    coll = db[coll_name]
                    try:
                        coll_info = coll.aggregate([{"$collStats": {"storageStats": {}}}]).next()
                        creation_time = coll_info.get('creationTime', None)
                        if creation_time and (oldest_time is None or creation_time < oldest_time):
                            oldest_time = creation_time
                    except OperationFailure:
                        # Skip collections where we can't get stats
                        continue

                if oldest_time:
                    print(f"Time created: {oldest_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print("Time created: Unknown")

                # Size
                size_mb = stats.get('dataSize', 0) / (1024 * 1024)
                print(f"Size: {size_mb:.2f} MB")

                # Additional specs
                print(f"Collections: {stats.get('collections', 'Unknown')}")
                print(f"Objects: {stats.get('objects', 'Unknown')}")
                print(f"Indexes: {stats.get('indexes', 'Unknown')}")

            except OperationFailure as e:
                print(f"Error getting stats for database {db_name}: {str(e)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <mongodb_ip> [port]")
        sys.exit(1)

    ip = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 27017

    get_mongo_specs(ip, port)