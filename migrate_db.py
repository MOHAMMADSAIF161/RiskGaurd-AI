import sqlite3
from pathlib import Path


DATABASE_FILE = (
    Path(__file__).resolve().parent
    / "riskguard.db"
)


SHAP_COLUMNS = {
    "shap_explanations": "JSON",
    "top_risk_factors": "JSON",
    "top_safe_factors": "JSON",
    "risk_explanation": "JSON",
}


def migrate():

    print("===================================")
    print("RiskGuard AI Database Migration")
    print("===================================")

    if not DATABASE_FILE.exists():

        print(
            f"Database not found: {DATABASE_FILE}"
        )

        return

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    try:

        # Check transactions table
        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name='transactions'
            """
        )

        if cursor.fetchone() is None:

            print(
                "transactions table does not exist."
            )

            return

        # Get existing columns
        cursor.execute(
            "PRAGMA table_info(transactions)"
        )

        existing_columns = {
            row[1]
            for row in cursor.fetchall()
        }

        print("\nChecking SHAP columns...\n")

        added = 0

        for column_name, column_type in SHAP_COLUMNS.items():

            if column_name in existing_columns:

                print(
                    f"✓ {column_name} already exists"
                )

            else:

                cursor.execute(
                    f"""
                    ALTER TABLE transactions
                    ADD COLUMN {column_name}
                    {column_type}
                    """
                )

                print(
                    f"+ Added {column_name}"
                )

                added += 1

        connection.commit()

        # Verify
        cursor.execute(
            "PRAGMA table_info(transactions)"
        )

        final_columns = {
            row[1]
            for row in cursor.fetchall()
        }

        print("\nSHAP column verification:\n")

        for column_name in SHAP_COLUMNS:

            if column_name in final_columns:

                print(
                    f"✓ {column_name}"
                )

            else:

                print(
                    f"✗ {column_name} MISSING"
                )

        # Verify transactions weren't deleted
        cursor.execute(
            "SELECT COUNT(*) FROM transactions"
        )

        transaction_count = (
            cursor.fetchone()[0]
        )

        print(
            "\nTransactions preserved:",
            transaction_count
        )

        print(
            "\nColumns added:",
            added
        )

        print(
            "\nSTATUS: DATABASE READY"
        )

    except Exception as exc:

        connection.rollback()

        print(
            "\nMigration failed:"
        )

        print(
            repr(exc)
        )

        raise

    finally:

        connection.close()

    print("===================================")


if __name__ == "__main__":
    migrate()