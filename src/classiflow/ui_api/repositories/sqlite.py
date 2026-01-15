"""SQLite repository for comments and reviews."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from classiflow.ui_api.models import (
    Comment,
    CommentCreate,
    Review,
    ReviewCreate,
    ReviewStatus,
)
from classiflow.ui_api.repositories.interfaces import (
    CommentRepository,
    ReviewRepository,
)

logger = logging.getLogger(__name__)

# SQL schema for tables
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    author TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_comments_scope ON comments(scope_type, scope_id);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reviews_scope ON reviews(scope_type, scope_id);
"""


class SQLiteCommentReviewRepository(CommentRepository, ReviewRepository):
    """
    SQLite-based repository for comments and reviews.

    Provides persistence for collaboration features in local mode.
    """

    def __init__(self, db_path: Path):
        """
        Initialize repository.

        Parameters
        ----------
        db_path : Path
            Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_db()

    def _ensure_db(self):
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        logger.info(f"SQLite database ready at {self.db_path}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # CommentRepository implementation
    # -------------------------------------------------------------------------

    def list_comments(
        self,
        scope_type: str,
        scope_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Comment], int]:
        """List comments for a scope."""
        with self._get_connection() as conn:
            # Get total count
            cursor = conn.execute(
                "SELECT COUNT(*) FROM comments WHERE scope_type = ? AND scope_id = ?",
                (scope_type, scope_id),
            )
            total = cursor.fetchone()[0]

            # Get paginated results
            offset = (page - 1) * page_size
            cursor = conn.execute(
                """
                SELECT id, scope_type, scope_id, author, content, created_at, updated_at
                FROM comments
                WHERE scope_type = ? AND scope_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (scope_type, scope_id, page_size, offset),
            )

            comments = [self._row_to_comment(row) for row in cursor.fetchall()]
            return comments, total

    def create_comment(self, data: CommentCreate) -> Comment:
        """Create a new comment."""
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO comments (scope_type, scope_id, author, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (data.scope_type, data.scope_id, data.author, data.content, now),
            )
            conn.commit()
            comment_id = cursor.lastrowid

            return Comment(
                id=comment_id,
                scope_type=data.scope_type,
                scope_id=data.scope_id,
                author=data.author,
                content=data.content,
                created_at=datetime.fromisoformat(now),
                updated_at=None,
            )

    def get_comment(self, comment_id: int) -> Optional[Comment]:
        """Get comment by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, scope_type, scope_id, author, content, created_at, updated_at
                FROM comments
                WHERE id = ?
                """,
                (comment_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_comment(row)
            return None

    def delete_comment(self, comment_id: int) -> bool:
        """Delete a comment."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM comments WHERE id = ?",
                (comment_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_comment(self, row: sqlite3.Row) -> Comment:
        """Convert database row to Comment model."""
        created_at = datetime.fromisoformat(row["created_at"])
        updated_at = None
        if row["updated_at"]:
            updated_at = datetime.fromisoformat(row["updated_at"])

        return Comment(
            id=row["id"],
            scope_type=row["scope_type"],
            scope_id=row["scope_id"],
            author=row["author"],
            content=row["content"],
            created_at=created_at,
            updated_at=updated_at,
        )

    # -------------------------------------------------------------------------
    # ReviewRepository implementation
    # -------------------------------------------------------------------------

    def list_reviews(
        self,
        scope_type: str,
        scope_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Review], int]:
        """List reviews for a scope."""
        with self._get_connection() as conn:
            # Get total count
            cursor = conn.execute(
                "SELECT COUNT(*) FROM reviews WHERE scope_type = ? AND scope_id = ?",
                (scope_type, scope_id),
            )
            total = cursor.fetchone()[0]

            # Get paginated results
            offset = (page - 1) * page_size
            cursor = conn.execute(
                """
                SELECT id, scope_type, scope_id, reviewer, status, notes, created_at, updated_at
                FROM reviews
                WHERE scope_type = ? AND scope_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (scope_type, scope_id, page_size, offset),
            )

            reviews = [self._row_to_review(row) for row in cursor.fetchall()]
            return reviews, total

    def create_review(self, data: ReviewCreate) -> Review:
        """Create a new review."""
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reviews (scope_type, scope_id, reviewer, status, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (data.scope_type, data.scope_id, data.reviewer, data.status.value, data.notes, now),
            )
            conn.commit()
            review_id = cursor.lastrowid

            return Review(
                id=review_id,
                scope_type=data.scope_type,
                scope_id=data.scope_id,
                reviewer=data.reviewer,
                status=data.status,
                notes=data.notes,
                created_at=datetime.fromisoformat(now),
                updated_at=None,
            )

    def get_review(self, review_id: int) -> Optional[Review]:
        """Get review by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, scope_type, scope_id, reviewer, status, notes, created_at, updated_at
                FROM reviews
                WHERE id = ?
                """,
                (review_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_review(row)
            return None

    def update_review_status(
        self,
        review_id: int,
        status: ReviewStatus,
        notes: Optional[str] = None,
    ) -> Optional[Review]:
        """Update review status."""
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            if notes is not None:
                cursor = conn.execute(
                    """
                    UPDATE reviews
                    SET status = ?, notes = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status.value, notes, now, review_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE reviews
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status.value, now, review_id),
                )
            conn.commit()

            if cursor.rowcount > 0:
                return self.get_review(review_id)
            return None

    def get_latest_review(
        self,
        scope_type: str,
        scope_id: str,
    ) -> Optional[Review]:
        """Get the most recent review for a scope."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, scope_type, scope_id, reviewer, status, notes, created_at, updated_at
                FROM reviews
                WHERE scope_type = ? AND scope_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (scope_type, scope_id),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_review(row)
            return None

    def _row_to_review(self, row: sqlite3.Row) -> Review:
        """Convert database row to Review model."""
        created_at = datetime.fromisoformat(row["created_at"])
        updated_at = None
        if row["updated_at"]:
            updated_at = datetime.fromisoformat(row["updated_at"])

        return Review(
            id=row["id"],
            scope_type=row["scope_type"],
            scope_id=row["scope_id"],
            reviewer=row["reviewer"],
            status=ReviewStatus(row["status"]),
            notes=row["notes"],
            created_at=created_at,
            updated_at=updated_at,
        )
