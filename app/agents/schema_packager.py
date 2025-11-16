"""
Schema Packager: Collect full schema metadata for selected tables.
Stage 4 of the pipeline.
"""
from typing import List, Dict, Any

from app.utils.logger import setup_logger

logger = setup_logger("schema_packager")


class SchemaPackager:
    """Agent for packaging complete schema metadata for SQL generation."""

    def __init__(self):
        """Initialize Schema Packager."""
        pass

    def package_schemas(
        self,
        selection: Dict[str, Any],
        query: str
    ) -> str:
        """
        Package complete schema metadata for selected tables.

        Args:
            selection: Table selection from Table Selector
            query: Original natural language query

        Returns:
            Formatted schema package string for SQL Generator
        """
        tables = selection["selected_tables"]
        joins = selection.get("joins", [])
        tier = selection["tier"]

        logger.info(f"[SCHEMA PACKAGER INPUT] Packaging schemas for {len(tables)} tables (tier {tier})")
        for idx, t in enumerate(tables, 1):
            has_columns = 'columns' in t and t['columns']
            logger.info(f"[INPUT TABLE {idx}] {t.get('table_name')}: has_columns={has_columns}, columns_count={len(t.get('columns', []))}")
            if has_columns:
                logger.debug(f"[INPUT TABLE {idx} COLUMNS] {[col['name'] for col in t['columns']]}")

        # Build schema package
        package_parts = []

        # Header
        package_parts.append("=" * 60)
        package_parts.append(f"QUERY: {query}")
        package_parts.append("=" * 60)
        package_parts.append("")
        package_parts.append(f"SELECTED TABLES: {len(tables)}")
        package_parts.append(f"TIER: {tier}")
        package_parts.append(f"CONFIDENCE: {selection.get('confidence', 0):.3f}")
        package_parts.append("")

        # Table schemas
        for idx, table_meta in enumerate(tables, 1):
            table_name = table_meta["table_name"]
            package_parts.append("=" * 60)
            package_parts.append(f"TABLE {idx}: {table_name}")
            package_parts.append("=" * 60)
            package_parts.append("")

            # Use schema from table metadata (already fetched from MSSQL)
            if 'columns' in table_meta:
                columns = table_meta['columns']
                package_parts.append("Columns:")
                for col in columns:
                    col_info = f"  • {col['name']} ({col['type']})"
                    if col.get('is_primary_key'):
                        col_info += " [PRIMARY KEY]"
                    if col.get('is_foreign_key'):
                        col_info += f" [FK -> {col.get('references')}]"
                    if not col.get('nullable', True):
                        col_info += " [NOT NULL]"
                    if col.get('max_length'):
                        col_info += f" [MAX: {col['max_length']}]"
                    package_parts.append(col_info)
                package_parts.append("")

                # Add row count if available
                if 'row_count' in table_meta:
                    package_parts.append(f"Approximate Row Count: {table_meta['row_count']:,}")
                    package_parts.append("")
            else:
                package_parts.append(f"[Schema not available]")
                package_parts.append("")

        # Joins
        if joins:
            package_parts.append("=" * 60)
            package_parts.append(f"JOINS ({selection.get('join_pattern', 'unknown')} pattern)")
            package_parts.append("=" * 60)
            package_parts.append("")

            for idx, join in enumerate(joins, 1):
                package_parts.append(f"JOIN {idx}:")
                package_parts.append(f"  {join['condition']}")
                package_parts.append(f"  Type: {join['type'].upper()}")
                if 'confidence' in join:
                    package_parts.append(f"  Confidence: {join['confidence']*100:.0f}%")
                package_parts.append("")

        # Instructions
        package_parts.append("=" * 60)
        package_parts.append("INSTRUCTIONS FOR SQL GENERATION")
        package_parts.append("=" * 60)
        package_parts.append("")
        package_parts.append("SQL Rules:")
        package_parts.append("  - Use Microsoft SQL Server (T-SQL) syntax")
        package_parts.append("  - Use SELECT TOP N instead of LIMIT for row limits")
        package_parts.append("  - Use aliases (t1, t2, t3) for tables")
        package_parts.append("  - Use ONLY the column names shown in the schema above")
        package_parts.append("  - Return only SQL query, no explanation")
        package_parts.append("")

        schema_package = "\n".join(package_parts)

        logger.info(f"[SCHEMA PACKAGER OUTPUT] Package created: {len(schema_package)} characters")
        logger.debug(f"[SCHEMA PACKAGE PREVIEW] First 500 chars:\n{schema_package[:500]}")
        return schema_package


def create_schema_packager() -> SchemaPackager:
    """Factory function to create Schema Packager instance."""
    return SchemaPackager()
