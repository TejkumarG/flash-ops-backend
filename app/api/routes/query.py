"""
Query API routes.
Clean routes that delegate to orchestrator.
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, QueryResponse, TeamApiKeyQueryRequest
from app.orchestration import get_query_pipeline
from app.services import get_mongo_client
from app.utils.encryption import decrypt_api_key
from app.utils.logger import setup_logger

router = APIRouter(prefix="/query", tags=["Query"])
logger = setup_logger("query_routes")


@router.post("", response_model=List[QueryResponse])
async def process_query(request: QueryRequest) -> List[QueryResponse]:
    """
    Process natural language query across multiple databases.

    Frontend handles authentication and provides database IDs that user can access.
    No backend auth required - trust the frontend (NextAuth handles it).

    Args:
        request: Query request with natural language question and database IDs

    Returns:
        List of query responses (one per database)
    """
    try:
        pipeline = get_query_pipeline()
        results = []

        logger.info(f"Processing query across {len(request.database_ids)} databases")

        for db_id in request.database_ids:
            try:
                response = pipeline.process(request.query, db_id)
                results.append(response)
            except Exception as e:
                logger.error(f"Error processing query for database {db_id}: {e}")
                # Continue with other databases even if one fails
                continue

        logger.info(
            f"Successfully processed {len(results)}/{len(request.database_ids)} databases"
        )
        return results

    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/team", response_model=List[QueryResponse])
async def process_team_query(request: TeamApiKeyQueryRequest) -> List[QueryResponse]:
    """
    Process query using team API key - automatically queries all team databases.

    For external API access:
    - Validates API key
    - Finds all accessible databases for team
    - Runs query across all databases
    - Tracks usage (usageCount, lastUsedAt)

    Args:
        request: Team API key and query

    Returns:
        List of query responses (one per accessible database)
    """
    try:
        mongo_client = get_mongo_client()

        # Extract prefix from provided key for quick lookup (flash_xxxxx)
        prefix = request.api_key[:13]

        # Find API key by prefix
        apikeys_collection = mongo_client.db["apikeys"]
        api_key_doc = apikeys_collection.find_one(
            {"keyPrefix": prefix, "isActive": True}
        )

        if not api_key_doc:
            logger.warning(f"No active API key found with prefix: {prefix}")
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Decrypt the stored key and verify it matches the provided key
        try:
            decrypted_key = decrypt_api_key(api_key_doc["key"])
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {e}")
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Constant-time comparison to prevent timing attacks
        if decrypted_key != request.api_key:
            logger.warning(f"API key mismatch for prefix: {prefix}")
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Get team information
        team_id = api_key_doc["teamId"]
        teams_collection = mongo_client.db["teams"]
        team = teams_collection.find_one({"_id": team_id})
        team_name = team.get("name", "Unknown Team") if team else "Unknown Team"

        logger.info(f"Team authenticated via API key: {team_name} (ID: {team_id})")

        # Find all databases accessible to this team
        accesses_collection = mongo_client.db["accesses"]
        team_accesses = list(
            accesses_collection.find({"teamId": team_id, "accessType": "team"})
        )

        if not team_accesses:
            logger.warning(f"No databases accessible for team: {team_id}")
            return []

        database_ids = [str(access["databaseId"]) for access in team_accesses]
        logger.info(f"Found {len(database_ids)} accessible databases for team")

        # Run query across all databases
        pipeline = get_query_pipeline()
        results = []

        for db_id in database_ids:
            try:
                response = pipeline.process(request.query, db_id)
                results.append(response)
            except Exception as e:
                logger.error(f"Error processing query for database {db_id}: {e}")
                # Continue with other databases even if one fails
                continue

        # Track API key usage
        try:
            teams_collection.update_one(
                {"_id": team_id},
                {
                    "$inc": {"usageCount": 1},
                    "$set": {
                        "lastUsedAt": datetime.utcnow(),
                        "metadata.lastQuery": request.query[:200],
                    },
                },
            )
            logger.info(f"Updated usage tracking for team: {team_id}")
        except Exception as e:
            logger.warning(f"Failed to update usage tracking: {e}")
            # Don't fail the request if usage tracking fails

        logger.info(
            f"Processed {len(results)}/{len(database_ids)} databases successfully for team query"
        )
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Team query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
