"""
SQL Generator: Generate SQL from natural language using LLM.
Stage 5 of the pipeline (Temporary - will be replaced).
"""
from typing import Optional
from openai import OpenAI

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("sql_generator")


class SQLGenerator:
    """Agent for generating SQL queries using LLM (temporary)."""

    def __init__(self):
        """Initialize SQL Generator."""
        self.client = None
        if settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate_sql(
        self,
        schema_package: str,
        attempt: int = 1,
        previous_errors: Optional[list] = None
    ) -> str:
        """
        Generate SQL query from schema package.

        Args:
            schema_package: Formatted schema from Schema Packager
            attempt: Attempt number (1-3)
            previous_errors: List of previous error messages

        Returns:
            Generated SQL query string
        """
        if not self.client:
            logger.error("OpenAI client not initialized (missing API key)")
            raise ValueError("OpenAI API key not configured")

        logger.info(f"Generating SQL (attempt {attempt}/{settings.MAX_REFLECTION_ATTEMPTS})")

        # Calculate temperature
        temperature = settings.TEMPERATURE_START + (attempt - 1) * settings.TEMPERATURE_INCREMENT

        # Build prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_user_prompt(schema_package, previous_errors)

        # Log schema package being sent to LLM
        logger.info(f"Schema package being sent to LLM:\n{schema_package}")

        try:
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )

            sql = response.choices[0].message.content.strip()

            # Clean SQL
            sql = self._clean_sql(sql)

            logger.info(f"SQL generated successfully (length: {len(sql)})")
            logger.info(f"Generated SQL: {sql}")

            return sql

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise

    def _get_system_prompt(self) -> str:
        """Get system prompt for SQL generation."""
        return """You are an expert SQL developer. Generate Microsoft SQL Server (T-SQL) queries based on the provided schema and natural language question.

You may receive multiple table combination options. Analyze each option and choose the BEST one for the query.

Rules:
- Return ONLY the SQL query, no explanations
- Use T-SQL/MSSQL syntax (SELECT TOP N instead of LIMIT, etc.)
- Use proper table aliases (t1, t2, t3)
- Refer to the provided schema for exact column names and types
- Handle date filters appropriately
- Use proper JOINs as specified in the schema
- Generate efficient, readable SQL
- If multiple options are provided, choose the one that best answers the query
- On retry attempts, try a DIFFERENT option than before"""

    def _build_user_prompt(
        self,
        schema_package: str,
        previous_errors: Optional[list]
    ) -> str:
        """Build user prompt with schema and errors."""
        parts = [schema_package]

        if previous_errors:
            parts.append("\n" + "=" * 60)
            parts.append("PREVIOUS ATTEMPT FAILED - TRY A DIFFERENT OPTION")
            parts.append("=" * 60)
            parts.append("")
            parts.append("Previous errors:")
            for idx, error in enumerate(previous_errors, 1):
                parts.append(f"  {idx}. {error}")
            parts.append("")
            parts.append("IMPORTANT: The previous combination didn't work.")
            parts.append("Choose a DIFFERENT option from the ones listed above.")
            parts.append("Analyze which option is more likely to succeed based on the error.")
            parts.append("")

        parts.append("Generate SQL:")

        return "\n".join(parts)

    def _clean_sql(self, sql: str) -> str:
        """Clean generated SQL."""
        # Remove markdown code blocks
        sql = sql.replace("```sql", "").replace("```", "")

        # Remove leading/trailing whitespace
        sql = sql.strip()

        # Remove trailing semicolon
        sql = sql.rstrip(';')

        return sql


def create_sql_generator() -> SQLGenerator:
    """Factory function to create SQL Generator instance."""
    return SQLGenerator()
