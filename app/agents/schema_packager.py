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

    def package_multiple_schemas(
        self,
        combinations: List[Dict[str, Any]],
        query: str
    ) -> str:
        """
        Package multiple table combinations into a single prompt for LLM to choose from.

        Args:
            combinations: List of table combinations (primary + alternatives)
            query: Original natural language query

        Returns:
            Formatted schema package string with all combinations
        """
        logger.info(f"[MULTI-SCHEMA PACKAGER] Packaging {len(combinations)} combinations in single prompt")

        package_parts = []

        # Header
        package_parts.append("=" * 60)
        package_parts.append(f"QUERY: {query}")
        package_parts.append("=" * 60)
        package_parts.append("")
        package_parts.append(f"You have {len(combinations)} table combination options below.")
        package_parts.append("ANALYZE ALL OPTIONS and DECIDE which one will best answer this query.")
        package_parts.append("Use your judgment - the confidence scores are just suggestions.")
        package_parts.append("")

        # Format each combination
        for combo_idx, selection in enumerate(combinations, 1):
            tables = selection["selected_tables"]
            joins = selection.get("joins", [])
            tier = selection["tier"]
            confidence = selection.get("confidence", 0)

            package_parts.append("#" * 60)
            package_parts.append(f"OPTION {combo_idx} (Confidence: {confidence:.3f}, Tier: {tier})")
            package_parts.append("#" * 60)
            package_parts.append("")

            # List table names for quick overview
            table_names = [t["table_name"] for t in tables]
            package_parts.append(f"Tables: {', '.join(table_names)}")
            package_parts.append(f"Pattern: {selection.get('join_pattern', 'unknown')}")
            package_parts.append("")

            # Table schemas
            for idx, table_meta in enumerate(tables, 1):
                table_name = table_meta["table_name"]
                package_parts.append("-" * 60)
                package_parts.append(f"TABLE {idx}: {table_name}")
                package_parts.append("-" * 60)

                # Use schema from table metadata
                if 'columns' in table_meta:
                    columns = table_meta['columns']
                    package_parts.append("Columns:")
                    for col in columns:
                        col_info = f"  • {col['name']} ({col['type']})"
                        if col.get('is_primary_key'):
                            col_info += " [PK]"
                        if col.get('is_foreign_key'):
                            col_info += f" [FK -> {col.get('references')}]"
                        if not col.get('nullable', True):
                            col_info += " [NOT NULL]"
                        package_parts.append(col_info)
                    package_parts.append("")

                    # Add row count
                    if 'row_count' in table_meta:
                        package_parts.append(f"Rows: ~{table_meta['row_count']:,}")
                        package_parts.append("")
                else:
                    package_parts.append("[Schema not available]")
                    package_parts.append("")

            # Joins for this combination
            if joins:
                package_parts.append("JOINS:")
                for idx, join in enumerate(joins, 1):
                    package_parts.append(f"  {idx}. {join['condition']} (Type: {join['type'].upper()})")
                package_parts.append("")

            package_parts.append("")

        # Final instructions
        package_parts.append("=" * 60)
        package_parts.append("INSTRUCTIONS")
        package_parts.append("=" * 60)
        package_parts.append("")
        package_parts.append("YOUR TASK:")
        package_parts.append("1. Carefully analyze the user's query and understand what they're asking for")
        package_parts.append("2. Review ALL table combination options provided above")
        package_parts.append("3. DECIDE FOR YOURSELF which combination will best answer the query")
        package_parts.append("   - Consider table relevance, available columns, and join relationships")
        package_parts.append("   - Don't just pick the first option - think about which makes most sense")
        package_parts.append("   - The confidence scores are hints, but use your own judgment")
        package_parts.append("4. Generate SQL using ONLY the tables and columns from your chosen option")
        package_parts.append("")
        package_parts.append("IMPORTANT - JOINs are OPTIONAL:")
        package_parts.append("  - If the query can be answered using ONE table, use only that table")
        package_parts.append("  - Only add JOINs if you actually need columns from multiple tables")
        package_parts.append("  - DO NOT join tables just because they're in the combination")
        package_parts.append("  - Unnecessary JOINs slow down queries and may return wrong results")
        package_parts.append("")
        package_parts.append("SQL Rules:")
        package_parts.append("  - Use Microsoft SQL Server (T-SQL) syntax")
        package_parts.append("  - Use SELECT TOP N instead of LIMIT")
        package_parts.append("  - Use table aliases (t1, t2, t3) only if you're using multiple tables")
        package_parts.append("  - Use ONLY columns that exist in your chosen option's schema")
        package_parts.append("  - Return ONLY the SQL query, no explanation or comments")
        package_parts.append("")

        schema_package = "\n".join(package_parts)

        logger.info(f"[MULTI-SCHEMA PACKAGER OUTPUT] Package created: {len(schema_package)} characters")
        return schema_package


def create_schema_packager() -> SchemaPackager:
    """Factory function to create Schema Packager instance."""
    return SchemaPackager()
