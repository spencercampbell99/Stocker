#!/bin/bash
set -e

# Wait for MySQL to be ready with retry limit
MAX_RETRY=30
RETRY_COUNT=0

until mysqladmin ping -h"localhost" --silent; do
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRY ]; then
        echo "Maximum retry attempts reached. MySQL may not be available."
        exit 1
    fi
    echo "Waiting for MySQL to be ready... (Attempt $RETRY_COUNT/$MAX_RETRY)"
    sleep 2
done

# Create the user and grant privileges
mysql -u root -p"${MYSQL_ROOT_PASSWORD}" << EOF
DROP USER IF EXISTS '${MYSQL_USER}'@'%';
CREATE USER IF NOT EXISTS '${MYSQL_USER}'@'${MYSQL_USER_HOST}' IDENTIFIED BY '${MYSQL_PASSWORD}';
GRANT ALL PRIVILEGES ON ${MYSQL_DATABASE}.* TO '${MYSQL_USER}'@'${MYSQL_USER_HOST}';
FLUSH PRIVILEGES;
EOF

echo "MySQL user '${MYSQL_USER}' created with access to database '${MYSQL_DATABASE}'"